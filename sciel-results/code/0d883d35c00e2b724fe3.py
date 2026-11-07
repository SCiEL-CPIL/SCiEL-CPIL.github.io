"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _check_triangle_free(n, adj):
    for u in range(n):
        neigh_u = adj[u]
        for v in neigh_u:
            if v <= u:
                continue
            if neigh_u & adj[v]:
                return False
    return True

def _bfs_diameter(n, adj):
    max_dist = 0
    for src in range(n):
        dist = [-1] * n
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    if dist[v] > max_dist:
                        max_dist = dist[v]
                    q.append(v)
    return max_dist

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = P['n']
    base_edges = set(tuple(sorted(e)) for e in P['edges'])
    all_possible = [(u, v) for u in range(n) for v in range(u + 1, n) if (u, v) not in base_edges]

    best_result = None
    best_count = float('inf')

    # Enumerate jump-length combinations
    jump_sets = []
    for a in range(3, 12):
        for b in range(a + 1, 13):
            jump_sets.append((a, b))
            for c in range(b + 1, 14):
                jump_sets.append((a, b, c))

    for jumps in jump_sets:
        if time.time() > deadline:
            break
        edges = set()
        for jump in jumps:
            for u in range(n - jump):
                edges.add((u, u + jump))
        edge_list = list(edges)
        adj = _build_adj(n, edge_list)
        if not _check_triangle_free(n, adj):
            continue
        diam = _bfs_diameter(n, adj)
        if diam <= 4 and len(edge_list) < best_count:
            best_count = len(edge_list)
            best_result = edge_list[:]

    if best_result is not None:
        return [[int(u), int(v)] for u, v in best_result]

    # Fallback: complete bipartite on parity classes
    return [[u, v] for u in range(n) for v in range(u + 1, n) if (u - v) % 2 and (u, v) not in base_edges]
# EVOLVE-BLOCK-END
