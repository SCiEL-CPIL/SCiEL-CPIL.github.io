"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s budget, 5s safety margin

    n = P['n']
    base_edges = set()
    for e in P['edges']:
        base_edges.add((min(e), max(e)))

    adj_base = [set() for _ in range(n)]
    for u, v in base_edges:
        adj_base[u].add(v)
        adj_base[v].add(u)

    def compute_diameter(adj):
        max_dist = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nbr in adj[node]:
                    if dist[nbr] == -1:
                        dist[nbr] = dist[node] + 1
                        queue.append(nbr)
            if -1 in dist:
                return 10**9
            md = max(dist)
            if md > max_dist:
                max_dist = md
        return max_dist

    def is_triangle_free_add(adj, u, v):
        # u,v not adjacent yet; check no common neighbor
        return len(adj[u] & adj[v]) == 0

    def check_valid(added_edges):
        adj = [set(s) for s in adj_base]
        for u, v in added_edges:
            if u == v:
                return False
            if v in adj[u]:  # already edge (base or added)
                return False
            if not is_triangle_free_add(adj, u, v):
                return False
            adj[u].add(v)
            adj[v].add(u)
        # connected + diameter
        dist = [-1] * n
        dist[0] = 0
        queue = deque([0])
        while queue:
            node = queue.popleft()
            for nbr in adj[node]:
                if dist[nbr] == -1:
                    dist[nbr] = dist[node] + 1
                    queue.append(nbr)
        if any(d == -1 for d in dist):
            return False
        return compute_diameter(adj) <= 4

    def try_remove_edges(current):
        # Greedily remove any edge whose removal keeps validity
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline - 3:
                    break
                trial = current[:i] + current[i+1:]
                if check_valid(trial):
                    current = trial
                    improved = True
                    break
        return current

    best_edges = None
    best_count = 10**9

    # Structured bipartite chord patterns: only even-odd edges (guaranteed triangle-free)
    # Pattern A: (i, i+k) for i where i+k < n, k odd (parity flip)
    for k in range(3, n, 2):  # odd k => parity flip => bipartite-safe
        if time.time() > deadline - 10:
            break
        for offset in range(k):
            edges_to_add = []
            seen = set()
            for i in range(offset, n - k):
                j = i + k
                key = (min(i, j), max(i, j))
                if key not in base_edges and key not in seen:
                    seen.add(key)
                    edges_to_add.append((i, j))
            if not edges_to_add:
                continue
            if check_valid(edges_to_add):
                edges_to_add = try_remove_edges([list(e) for e in edges_to_add])
                if len(edges_to_add) < best_count:
                    best_count = len(edges_to_add)
                    best_edges = [list(e) for e in edges_to_add]

    # Pattern B: hub-based bipartite. Pick hub sets in L (even) and R (odd).
    # Connect each non-hub to nearest hub in opposite part (bipartite-safe).
    L = [i for i in range(n) if i % 2 == 0]
    R = [i for i in range(n) if i % 2 == 1]
    if time.time() < deadline - 10:
        for hL in range(1, 5):
            for hR in range(1, 5):
                if time.time() > deadline - 10:
                    break
                # Choose evenly spaced hubs
                stepL = len(L) // hL
                hubsL = [L[min(i * stepL, len(L) - 1)] for i in range(hL)]
                stepR = len(R) // hR
                hubsR = [R[min(i * stepR, len(R) - 1)] for i in range(hR)]
                edges_to_add = []
                seen = set()
                # Connect each L vertex to a hub in R (bipartite: L-R)
                for v in L:
                    best_hub = min(hubsR, key=lambda h: abs(h - v))
                    key = (min(v, best_hub), max(v, best_hub))
                    if key not in base_edges and key not in seen:
                        seen.add(key)
                        edges_to_add.append((v, best_hub))
                # Connect each R vertex to a hub in L
                for v in R:
                    best_hub = min(hubsL, key=lambda h: abs(h - v))
                    key = (min(v, best_hub), max(v, best_hub))
                    if key not in base_edges and key not in seen:
                        seen.add(key)
                        edges_to_add.append((v, best_hub))
                if check_valid(edges_to_add):
                    edges_to_add = try_remove_edges([list(e) for e in edges_to_add])
                    if len(edges_to_add) < best_count:
                        best_count = len(edges_to_add)
                        best_edges = [list(e) for e in edges_to_add]

    # Pattern C: full bipartite between two small sets (complete bipartite K_{a,b} is triangle-free)
    if time.time() < deadline - 10:
        for a in range(2, 7):
            for b in range(2, 7):
                if time.time() > deadline - 10:
                    break
                # Pick a vertices from L, b from R, spread out
                stepL = len(L) // a
                setL = [L[min(i * stepL, len(L) - 1)] for i in range(a)]
                stepR = len(R) // b
                setR = [R[min(i * stepR, len(R) - 1)] for i in range(b)]
                edges_to_add = []
                seen = set()
                for u in setL:
                    for v in setR:
                        key = (min(u, v), max(u, v))
                        if key not in base_edges and key not in seen:
                            seen.add(key)
                            edges_to_add.append((u, v))
                if check_valid(edges_to_add):
                    edges_to_add = try_remove_edges([list(e) for e in edges_to_add])
                    if len(edges_to_add) < best_count:
                        best_count = len(edges_to_add)
                        best_edges = [list(e) for e in edges_to_add]

    # Fallback: all even-odd non-path edges (definitely valid, triangle-free, connected)
    if best_edges is None:
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u % 2) != (v % 2) and (u, v) not in base_edges:
                    best_edges.append([u, v])
        best_edges = try_remove_edges(best_edges)

    # Final verification
    if not check_valid(best_edges):
        # Absolute fallback: dense bipartite
        best_edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (u % 2) != (v % 2) and (u, v) not in base_edges:
                    best_edges.append([u, v])

    return [[int(u), int(v)] for u, v in best_edges]

# EVOLVE-BLOCK-END
