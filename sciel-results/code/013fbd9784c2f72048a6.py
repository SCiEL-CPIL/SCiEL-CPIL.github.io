"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _all_4cycles(adj):
    n = len(adj)
    seen = set()
    for a in range(n):
        for b in range(a + 1, n):
            if b not in adj[a]:
                continue
            for c in range(b + 1, n):
                if c not in adj[b]:
                    continue
                for d in range(c + 1, n):
                    if d not in adj[c]:
                        continue
                    if a in adj[d]:
                        # cycle a-b-c-d-a
                        cyc = (a, b, c, d)
                        seen.add(cyc)
    return seen

def _is_rainbow(edges_dict, adj):
    # edges_dict: (u,v) -> colour, u<v
    for (a, b, c, d) in _all_4cycles(adj):
        cols = []
        for (x, y) in [(a, b), (b, c), (c, d), (d, a)]:
            u, v = (x, y) if x < y else (y, x)
            cols.append(edges_dict[(u, v)])
        if len(set(cols)) != 4:
            return False
    return True

def _check_valid(result):
    n = 20
    adj = [set() for _ in range(n)]
    edges_dict = {}
    for (u, v, c) in result:
        if u == v:
            return False
        if u > v:
            u, v = v, u
        if (u, v) in edges_dict:
            return False
        edges_dict[(u, v)] = c
        adj[u].add(v)
        adj[v].add(u)
    return _is_rainbow(edges_dict, adj)

def construct():
    deadline = time.time() + 120.0
    n = 20
    # 4 groups of 5
    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]]
    gidx = {}
    for gi, g in enumerate(groups):
        for v in g:
            gidx[v] = gi

    edges = {}  # (u,v) u<v -> colour
    # within-group: 5-cycle, colour = gi
    for gi, g in enumerate(groups):
        for i in range(5):
            u, v = g[i], g[(i + 1) % 5]
            a, b = (u, v) if u < v else (v, u)
            edges[(a, b)] = gi
    # between-group: matching, colour = 4 + pair_index
    pair_idx = {}
    pi = 0
    for i in range(4):
        for j in range(i + 1, 4):
            pair_idx[(i, j)] = pi
            pi += 1
    for i in range(4):
        for j in range(i + 1, 4):
            gi, gj = groups[i], groups[j]
            for k in range(5):
                u, v = gi[k], gj[k]
                a, b = (u, v) if u < v else (v, u)
                edges[(a, b)] = 4 + pair_idx[(i, j)]

    result = [[a, b, c] for (a, b), c in edges.items()]
    if not _check_valid(result):
        # fallback to star
        return [[0, v, 0] for v in range(1, 20)]

    # Greedily add missing edges if they keep validity, within deadline
    all_pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    for (a, b) in all_pairs:
        if time.time() > deadline - 20.0:
            break
        if (a, b) in edges:
            continue
        # try colours 0..19
        best_c = None
        for c in range(20):
            edges[(a, b)] = c
            if _check_valid([[x, y, z] for (x, y), z in edges.items()]):
                best_c = c
                break
        if best_c is not None:
            edges[(a, b)] = best_c
        else:
            del edges[(a, b)]

    result = [[a, b, c] for (a, b), c in edges.items()]
    if _check_valid(result):
        return result
    return [[0, v, 0] for v in range(1, 20)]
# EVOLVE-BLOCK-END
