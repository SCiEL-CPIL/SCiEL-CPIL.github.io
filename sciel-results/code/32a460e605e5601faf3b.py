import time
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Multi-start greedy edge addition to find a dense valid graph on 15 vertices
    where no 3-edge path has endpoints of equal degree.
    """
    n = 15
    deadline = time.time() + 78

    def build_state(edges):
        adj = [[False] * n for _ in range(n)]
        degree = [0] * n
        neigh = [set() for _ in range(n)]
        for u, v in edges:
            adj[u][v] = adj[v][u] = True
            degree[u] += 1
            degree[v] += 1
            neigh[u].add(v)
            neigh[v].add(u)
        return adj, degree, neigh

    def can_add(adj, degree, neigh, u, v):
        nd = degree[:]
        nd[u] += 1
        nd[v] += 1
        # Case B: p0-u-v-p3 (new edge is middle)
        for p0 in neigh[u]:
            if p0 == v:
                continue
            for p3 in neigh[v]:
                if p3 == u or p3 == p0:
                    continue
                if nd[p0] == nd[p3]:
                    return False
        # Case A: u-v-p2-p3 (new edge is first)
        for p2 in neigh[v]:
            if p2 == u:
                continue
            for p3 in neigh[p2]:
                if p3 in (u, v, p2):
                    continue
                if nd[u] == nd[p3]:
                    return False
        # Case C: p0-p1-v-u (new edge is last)
        for p1 in neigh[v]:
            if p1 == u:
                continue
            for p0 in neigh[p1]:
                if p0 in (u, v, p1):
                    continue
                if nd[p0] == nd[u]:
                    return False
        return True

    def greedy_from(order, adj0=None, deg0=None, neigh0=None):
        if adj0 is not None:
            adj = [row[:] for row in adj0]
            degree = deg0[:]
            neigh = [s.copy() for s in neigh0]
            edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if adj[i][j]:
                        edges.append([i, j])
        else:
            adj = [[False] * n for _ in range(n)]
            degree = [0] * n
            neigh = [set() for _ in range(n)]
            edges = []

        for u, v in order:
            if time.time() > deadline:
                break
            if adj[u][v]:
                continue
            if can_add(adj, degree, neigh, u, v):
                adj[u][v] = adj[v][u] = True
                degree[u] += 1
                degree[v] += 1
                neigh[u].add(v)
                neigh[v].add(u)
                edges.append([u, v])
        return edges

    all_edges = list(combinations(range(n), 2))
    best_edges = []

    # Start 1: greedy from empty, natural order
    edges = greedy_from(all_edges)
    if len(edges) > len(best_edges):
        best_edges = edges

    # Start 2: greedy from empty, reverse
    edges = greedy_from(list(reversed(all_edges)))
    if len(edges) > len(best_edges):
        best_edges = edges

    # Start 3: K_{7,8} base, then try adding remaining edges
    A = list(range(7))
    B = list(range(7, 15))
    base = [[u, v] for u in A for v in B]
    adj, deg, neigh = build_state(base)
    remaining = [(u, v) for u, v in all_edges if not adj[u][v]]
    edges = greedy_from(remaining, adj, deg, neigh)
    if len(edges) > len(best_edges):
        best_edges = edges

    # Random restarts from empty
    import random
    for seed in range(30):
        if time.time() > deadline - 5:
            break
        rng = random.Random(seed)
        shuffled = all_edges[:]
        rng.shuffle(shuffled)
        edges = greedy_from(shuffled)
        if len(edges) > len(best_edges):
            best_edges = edges

    return best_edges
# EVOLVE-BLOCK-END
