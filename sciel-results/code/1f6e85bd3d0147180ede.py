import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 110
    n = 20
    ncol = 20

    best_edges = []
    best_count = -1

    def key(a, b):
        return (a, b) if a < b else (b, a)

    def run_once(seed):
        nonlocal best_edges, best_count
        rng = random.Random(seed)
        adj = {}
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        edges = []
        neigh = {i: set() for i in range(n)}
        for (u, v) in all_edges:
            if time.time() > DEADLINE:
                break
            forbidden = set()
            for a in neigh[v]:
                if a == u:
                    continue
                for b in neigh[u]:
                    if b == v or b == a:
                        continue
                    if a in neigh[b]:
                        c1 = adj[key(v, a)]
                        c2 = adj[key(a, b)]
                        c3 = adj[key(b, u)]
                        forbidden.add(c1)
                        forbidden.add(c2)
                        forbidden.add(c3)
            if len(forbidden) < ncol:
                for c in range(ncol):
                    if c not in forbidden:
                        adj[key(u, v)] = c
                        edges.append([u, v, c])
                        neigh[u].add(v)
                        neigh[v].add(u)
                        break
        if len(edges) > best_count:
            best_count = len(edges)
            best_edges = edges[:]

    for seed in range(0, 500):
        if time.time() > DEADLINE - 3:
            break
        run_once(seed)

    return best_edges
