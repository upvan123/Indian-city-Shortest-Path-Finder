"""
============================================================
  INDIAN CITY SHORTEST PATH FINDER
  Design & Analysis of Algorithms — Project
  Algorithms: Dijkstra | BFS | DFS | A*
============================================================
"""

import heapq
import math
import time
from collections import deque

# ─────────────────────────────────────────────
#  CITY DATA  (name: (lat, lon))
# ─────────────────────────────────────────────
CITIES = {
    "Delhi":       (28.61, 77.21),
    "Mumbai":      (19.08, 72.88),
    "Kolkata":     (22.57, 88.36),
    "Chennai":     (13.08, 80.27),
    "Bangalore":   (12.97, 77.59),
    "Hyderabad":   (17.38, 78.49),
    "Pune":        (18.52, 73.86),
    "Ahmedabad":   (23.03, 72.59),
    "Jaipur":      (26.91, 75.79),
    "Lucknow":     (26.85, 80.95),
    "Nagpur":      (21.15, 79.09),
    "Surat":       (21.17, 72.83),
    "Bhopal":      (23.26, 77.41),
    "Patna":       (25.59, 85.14),
    "Chandigarh":  (30.73, 76.78),
    "Indore":      (22.72, 75.86),
    "Varanasi":    (25.32, 83.01),
    "Agra":        (27.18, 78.01),
    "Amritsar":    (31.63, 74.87),
    "Jodhpur":     (26.29, 73.02),
    "Kochi":       (9.93,  76.27),
    "Coimbatore":  (11.02, 76.97),
    "Visakhapatnam":(17.69, 83.22),
    "Bhubaneswar": (20.30, 85.82),
    "Guwahati":    (26.14, 91.74),
    "Ranchi":      (23.34, 85.31),
    "Jabalpur":    (23.18, 79.95),
    "Raipur":      (21.25, 81.63),
}

# ─────────────────────────────────────────────
#  EDGES  [cityA, cityB, distance_km]
# ─────────────────────────────────────────────
EDGES = [
    ("Delhi","Agra",206), ("Delhi","Jaipur",270), ("Delhi","Chandigarh",250),
    ("Delhi","Lucknow",555), ("Delhi","Bhopal",770), ("Delhi","Amritsar",450),
    ("Agra","Jaipur",235), ("Agra","Lucknow",340), ("Agra","Varanasi",570),
    ("Jaipur","Jodhpur",340), ("Jaipur","Ahmedabad",650), ("Jaipur","Bhopal",530),
    ("Chandigarh","Amritsar",220),
    ("Amritsar","Jodhpur",700),
    ("Jodhpur","Ahmedabad",460),
    ("Ahmedabad","Surat",270), ("Ahmedabad","Mumbai",530), ("Ahmedabad","Indore",390),
    ("Surat","Mumbai",310), ("Surat","Pune",390),
    ("Mumbai","Pune",150), ("Mumbai","Hyderabad",710), ("Mumbai","Nagpur",830),
    ("Pune","Hyderabad",570), ("Pune","Nagpur",710),
    ("Bhopal","Nagpur",360), ("Bhopal","Indore",195), ("Bhopal","Jabalpur",290),
    ("Indore","Nagpur",490),
    ("Jabalpur","Nagpur",310), ("Jabalpur","Raipur",440),
    ("Nagpur","Hyderabad",500), ("Nagpur","Raipur",300), ("Nagpur","Ranchi",700),
    ("Lucknow","Varanasi",285), ("Lucknow","Patna",490),
    ("Varanasi","Patna",220),
    ("Patna","Ranchi",330), ("Patna","Kolkata",610),
    ("Ranchi","Kolkata",400), ("Ranchi","Bhubaneswar",440), ("Ranchi","Raipur",340),
    ("Raipur","Bhubaneswar",430), ("Raipur","Visakhapatnam",600),
    ("Bhubaneswar","Kolkata",450), ("Bhubaneswar","Visakhapatnam",440),
    ("Kolkata","Guwahati",1000),
    ("Hyderabad","Chennai",630), ("Hyderabad","Bangalore",570), ("Hyderabad","Visakhapatnam",620),
    ("Chennai","Bangalore",350), ("Chennai","Coimbatore",500), ("Chennai","Kochi",680),
    ("Bangalore","Coimbatore",360), ("Bangalore","Kochi",530),
    ("Coimbatore","Kochi",200),
]

# Build adjacency list
def build_graph():
    graph = {c: [] for c in CITIES}
    for a, b, d in EDGES:
        graph[a].append((b, d))
        graph[b].append((a, d))
    return graph

GRAPH = build_graph()

# ─────────────────────────────────────────────
#  HEURISTIC  (Haversine distance in km)
# ─────────────────────────────────────────────
def haversine(c1, c2):
    lat1, lon1 = math.radians(CITIES[c1][0]), math.radians(CITIES[c1][1])
    lat2, lon2 = math.radians(CITIES[c2][0]), math.radians(CITIES[c2][1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 6371 * 2 * math.asin(math.sqrt(a))

def reconstruct_path(prev, src, dst):
    path, cur = [], dst
    while cur:
        path.append(cur)
        cur = prev.get(cur)
    path.reverse()
    return path if path and path[0] == src else []

def path_distance(path):
    total = 0
    for i in range(len(path)-1):
        for nb, d in GRAPH[path[i]]:
            if nb == path[i+1]:
                total += d
                break
    return total

# ─────────────────────────────────────────────
#  ALGORITHM 1: DIJKSTRA'S
# ─────────────────────────────────────────────
def dijkstra(src, dst):
    dist = {c: float('inf') for c in CITIES}
    dist[src] = 0
    prev = {}
    visited = set()
    pq = [(0, src)]
    nodes_explored = 0

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        nodes_explored += 1
        if u == dst:
            break
        for v, w in GRAPH[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))

    path = reconstruct_path(prev, src, dst)
    return path, dist[dst], nodes_explored

# ─────────────────────────────────────────────
#  ALGORITHM 2: BFS
# ─────────────────────────────────────────────
def bfs(src, dst):
    visited = {src}
    prev = {}
    queue = deque([src])
    nodes_explored = 0

    while queue:
        u = queue.popleft()
        nodes_explored += 1
        if u == dst:
            break
        for v, _ in GRAPH[u]:
            if v not in visited:
                visited.add(v)
                prev[v] = u
                queue.append(v)

    path = reconstruct_path(prev, src, dst)
    return path, path_distance(path), nodes_explored

# ─────────────────────────────────────────────
#  ALGORITHM 3: DFS
# ─────────────────────────────────────────────
def dfs(src, dst):
    visited = set()
    prev = {}
    nodes_explored = 0
    found = [False]

    def _dfs(u):
        if found[0]:
            return
        visited.add(u)
        nonlocal nodes_explored
        nodes_explored += 1
        if u == dst:
            found[0] = True
            return
        for v, _ in GRAPH[u]:
            if v not in visited:
                prev[v] = u
                _dfs(v)

    _dfs(src)
    path = reconstruct_path(prev, src, dst)
    return path, path_distance(path), nodes_explored

# ─────────────────────────────────────────────
#  ALGORITHM 4: A* SEARCH
# ─────────────────────────────────────────────
def astar(src, dst):
    g = {c: float('inf') for c in CITIES}
    g[src] = 0
    prev = {}
    open_set = [(haversine(src, dst), 0, src)]
    closed = set()
    nodes_explored = 0

    while open_set:
        f, cost, u = heapq.heappop(open_set)
        if u in closed:
            continue
        closed.add(u)
        nodes_explored += 1
        if u == dst:
            break
        for v, w in GRAPH[u]:
            if v in closed:
                continue
            tentative_g = g[u] + w
            if tentative_g < g[v]:
                g[v] = tentative_g
                prev[v] = u
                heapq.heappush(open_set, (tentative_g + haversine(v, dst), tentative_g, v))

    path = reconstruct_path(prev, src, dst)
    return path, g[dst], nodes_explored

# ─────────────────────────────────────────────
#  RUNNER  — compare all algorithms
# ─────────────────────────────────────────────
def compare_all(src, dst):
    algorithms = {
        "Dijkstra's": dijkstra,
        "BFS":        bfs,
        "DFS":        dfs,
        "A* Search":  astar,
    }
    complexities = {
        "Dijkstra's": ("O((V+E) log V)", "O(V)"),
        "BFS":        ("O(V+E)",         "O(V)"),
        "DFS":        ("O(V+E)",         "O(V)"),
        "A* Search":  ("O(E log V)",     "O(V)"),
    }
    results = {}
    print(f"\n{'='*60}")
    print(f"  Source: {src}  →  Destination: {dst}")
    print(f"{'='*60}")

    for name, fn in algorithms.items():
        t0 = time.perf_counter()
        path, dist, explored = fn(src, dst)
        elapsed = (time.perf_counter() - t0) * 1000
        results[name] = {
            "path": path,
            "distance_km": round(dist),
            "nodes_explored": explored,
            "path_length": len(path),
            "time_ms": round(elapsed, 4),
            "time_complexity": complexities[name][0],
            "space_complexity": complexities[name][1],
        }
        print(f"\n  [{name}]")
        print(f"    Path     : {' → '.join(path) if path else 'No path found'}")
        print(f"    Distance : {round(dist):,} km")
        print(f"    Nodes    : {explored} explored")
        print(f"    Time     : {elapsed:.4f} ms")

    # Comparison table
    print(f"\n{'─'*60}")
    print(f"  COMPARISON TABLE")
    print(f"{'─'*60}")
    header = f"  {'Algorithm':<16} {'Dist(km)':<12} {'Nodes':<10} {'Hops':<8} {'Time(ms)':<12}"
    print(header)
    print(f"  {'─'*56}")
    for name, r in results.items():
        best_dist = min(v['distance_km'] for v in results.values())
        flag = " ✓" if r['distance_km'] == best_dist else ""
        print(f"  {name:<16} {r['distance_km']:<12,} {r['nodes_explored']:<10} {r['path_length']:<8} {r['time_ms']:<12}{flag}")
    print(f"{'='*60}\n")
    return results

# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  INDIAN CITY SHORTEST PATH FINDER")
    print("  Design & Analysis of Algorithms — Project\n")
    print("  Available Cities:")
    for i, c in enumerate(sorted(CITIES), 1):
        print(f"    {i:2}. {c}")

    print("\n  Running sample: Delhi → Kochi")
    compare_all("Delhi", "Kochi")

    print("  Running sample: Amritsar → Guwahati")
    compare_all("Amritsar", "Guwahati")

    print("  Running sample: Mumbai → Kolkata")
    compare_all("Mumbai", "Kolkata")