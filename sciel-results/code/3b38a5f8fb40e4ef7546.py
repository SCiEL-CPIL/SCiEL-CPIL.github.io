"""Clique partition of two fixed dense random graphs via exact bitmask greedy."""
import time
import sys

sys.setrecursionlimit(100000)


def _gen(n, p, seed):
    from random import Random
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges


def _bit_count(x):
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c


def _max_clique(cands, adj, size, deadline):
    # Tomita branch-and-bound on bitmasks.
    best = []
    best_size = 0
    stack = [cands]
    while stack:
        cands = stack.pop()
        # bound
        if _bit_count(cands) + len(best) <= best_size:
            continue
        if time.time() > deadline:
            continue
        while cands:
            v = (cands & -cands).bit_length() - 1
            cands &= cands - 1
            nc = cands & adj[v]
            if _bit_count(nc) + len(best) + 1 <= best_size:
                continue
            if time.time() > deadline:
                continue
            # recurse with nc
            stack.append(nc)
            # We need to actually build the clique: use a DFS instead.
        # (placeholder; real DFS below)
    return best


def _solve(n, edges, deadline):
    # adjacency as bitmask
    adj = [0] * n
    for (u, v) in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    remaining = 0
    for (u, v) in edges:
        remaining |= (1 << u) | (1 << v)
    if remaining == 0:
        return []

    cliques = []

    def max_clique(cands):
        best = []
        best_size = 0

        def dfs(cands, cur):
            nonlocal best, best_size
            if time.time() > deadline:
                return
            if not cands:
                if len(cur) > best_size:
                    best_size = len(cur)
                    best = cur[:]
                return
            if len(cur) + _bit_count(cands) <= best_size:
                return
            # pivot
            v = (cands & -cands).bit_length() - 1
            pc = cands & adj[v]
            rest = cands
            while rest:
                x = (rest & -rest).bit_length() - 1
                rest &= rest - 1
                if pc & (1 << x):
                    continue
                # try x
                cur.append(x)
                dfs(cands & adj[x], cur)
                cur.pop()
                cands &= ~(1 << x)
                if len(cur) + _bit_count(cands) <= best_size:
                    return

        dfs(cands, [])
        return best

    while remaining:
        # pick edge with fewest common neighbours
        best_edge = None
        best_cnt = 10**9
        for (u, v) in edges:
            if remaining & (1 << u) and remaining & (1 << v):
                c = _bit_count(adj[u] & adj[v] & remaining)
                if c < best_cnt:
                    best_cnt = c
                    best_edge = (u, v)
                    if c == 0:
                        break
        u, v = best_edge
        cands = adj[u] & adj[v] & remaining
        cands |= (1 << u) | (1 << v)
        cl = max_clique(cands)
        if not cl:
            cl = [u, v]
        cliques.append(cl)
        mask = 0
        for x in cl:
            mask |= 1 << x
        remaining &= ~mask
        if time.time() > deadline:
            break

    # If deadline forced us to stop, fall back to trivial edges for remaining.
    if remaining:
        for (u, v) in edges:
            if remaining & (1 << u) and remaining & (1 << v):
                cliques.append([u, v])
                remaining &= ~(1 << u)
                remaining &= ~(1 << v)
    return cliques


def construct():
    t0 = time.time()
    total_budget = 120.0
    results = []
    for i, (n, p, seed) in enumerate(((32, 0.6, 601), (64, 0.55, 602))):
        edges = _gen(n, p, seed)
        # give more time to the larger instance
        if i == 0:
            dl = t0 + 60.0
        else:
            dl = t0 + total_budget - 5.0
        res = _solve(n, edges, dl)
        results.append(res)
    return results
