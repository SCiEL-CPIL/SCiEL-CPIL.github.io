import time
from collections import deque

def construct():
    """
    Finds minimum edges to add to path P_24 for triangle-free,
    connected graph with diameter <= 4, using greedy + local search.
    """
    start_time = time.time()
    deadline = start_time + 50

    n = 24
    path_edges = [(i, i + 1) for i in range(n - 1)]
    path_edge_set = set((min(u, v), max(u, v)) for u, v in path_edges)

    even_nodes = [i for i in range(n) if i % 2 == 0]
    odd_nodes = [i for i in range(n) if i % 2 == 1]

    candidates = []
    for e in even_nodes:
        for o in odd_nodes:
            key = (min(e, o), max(e, o))
            if key not in path_edge_set:
                candidates.append(key)

    def build_adj(added_edges):
        adj = [[] for _ in range(n)]
        for u, v in path_edges:
            adj[u].append(v)
            adj[v].append(u)
        for u, v in added_edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    def bfs_diameter(adj):
        max_d = 0
        for src in range(n):
            dist = [-1] * n
            dist[src] = 0
            q = deque([src])
            while q:
                node = q.popleft()
                for nb in adj[node]:
                    if dist[nb] == -1:
                        dist[nb] = dist[node] + 1
                        q.append(nb)
            if -1 in dist:
                return 999
            max_d = max(max_d, max(dist))
        return max_d

    def is_valid(added_edges):
        return bfs_diameter(build_adj(added_edges)) <= 4

    # Phase 1: Greedy - add edge that most reduces diameter
    added = []
    remaining = list(candidates)

    for _ in range(10):
        if time.time() > deadline - 15:
            break
        best_edge = None
        best_diam = 999
        for cand in remaining:
            test = added + [cand]
            d = bfs_diameter(build_adj(test))
            if d < best_diam:
                best_diam = d
                best_edge = cand
        if best_edge is None:
            break
        added.append(best_edge)
        remaining.remove(best_edge)
        if best_diam <= 4:
            break

    # Phase 2: Local search - try removing each edge
    improved = True
    while improved:
        improved = False
        for i in range(len(added)):
            if time.time() > deadline - 10:
                break
            test = added[:i] + added[i + 1:]
            if is_valid(test):
                added = test
                improved = True
                break

    # Phase 3: Verify and return
    if is_valid(added):
        result = [[int(u), int(v)] for u, v in added]
        return result

    # Fallback: known valid 9-edge hub solution
    hub = 12
    leaves = [0, 2, 4, 6, 8, 16, 18, 20, 22]
    added_edges = [[hub, v] for v in leaves]
    result = [[int(u), int(v)] for u, v in added_edges]
    return result
