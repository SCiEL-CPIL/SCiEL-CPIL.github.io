import numpy as np
import time
import random

def _score(adj, deg):
    # adj: 16x16 bool, deg: 16-int array
    n = adj.shape[0]
    best = -1
    for u in range(n):
        for v in range(u+1, n):
            if adj[u, v]:
                for w in range(v+1, n):
                    if adj[u, w] and adj[v, w]:
                        s = deg[u] + deg[v] + deg[w]
                        if s > best:
                            best = s
    return best if best >= 0 else 0

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin under 120s
    n = 16
    m = 65
    rng = random.Random(12345)

    all_pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    pair_index = {p: k for k, p in enumerate(all_pairs)}

    def make_seed():
        # K_{6,10} + 5 edges inside the size-10 part (a matching of 5)
        A = list(range(6))
        B = list(range(6, 16))
        edges = set()
        for u in A:
            for v in B:
                edges.add((u, v))
        # 5 matching edges inside B
        for k in range(5):
            edges.add((B[2*k], B[2*k+1]))
        return edges

    def edges_to_adj(edges):
        adj = np.zeros((n, n), dtype=bool)
        deg = np.zeros(n, dtype=int)
        for (u, v) in edges:
            adj[u, v] = adj[v, u] = True
            deg[u] += 1
            deg[v] += 1
        return adj, deg

    # Evaluate seeds, pick best
    best_edges = None
    best_score = 10**9
    for _ in range(20):
        e = make_seed()
        # randomize which 5 matching edges / which part gets extras
        adj, deg = edges_to_adj(e)
        s = _score(adj, deg)
        if s < best_score:
            best_score = s
            best_edges = e

    adj, deg = edges_to_adj(best_edges)
    cur_score = _score(adj, deg)
    best_score = cur_score
    best_edges = set(best_edges)

    T = 3.0
    T_min = 0.05
    cool = 0.9999

    iters = 0
    while time.time() < deadline:
        iters += 1
        # pick a random edge to remove and a random non-edge to add
        e_list = list(best_edges)
        u, v = e_list[rng.randrange(len(e_list))]
        # find a non-edge
        while True:
            a, b = all_pairs[rng.randrange(len(all_pairs))]
            if (a, b) not in best_edges and (b, a) not in best_edges:
                break
        new_edges = set(best_edges)
        new_edges.discard((u, v))
        new_edges.discard((v, u))
        new_edges.add((a, b))
        new_edges.add((b, a))
        # ensure exactly m edges (undirected count)
        # count undirected
        undir = set()
        for (x, y) in new_edges:
            undir.add((min(x, y), max(x, y)))
        if len(undir) != m:
            continue
        adj2, deg2 = edges_to_adj(undir)
        s2 = _score(adj2, deg2)
        if s2 <= cur_score or rng.random() < np.exp(-(s2 - cur_score) / max(T, 1e-9)):
            cur_score = s2
            best_edges = undir
            adj, deg = adj2, deg2
            if s2 < best_score:
                best_score = s2
        T *= cool
        if T < T_min:
            T = T_min

    # Final: return best_edges as list of [u,v]
    out = []
    for (u, v) in best_edges:
        if u < v:
            out.append([u, v])
    return {'n': 16, 'edges': out}
