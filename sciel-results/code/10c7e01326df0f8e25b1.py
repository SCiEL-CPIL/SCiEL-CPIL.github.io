import time
import random

def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)

def _norm(e):
    a, b = e
    return (a, b) if a <= b else (b, a)

def _cycle_edges(cycle):
    L = len(cycle)
    out = []
    for i in range(L):
        a = cycle[i]
        b = cycle[(i + 1) % L]
        out.append(_norm((a, b)))
    return out

def _rand_long_cycle(adj, n, rng, iters, deadline):
    best = []
    for _ in range(iters):
        if time.time() > deadline:
            break
        start = rng.randrange(n)
        path = [start]
        on = {start}
        u = start
        while True:
            if start in adj[u] and len(path) >= 3:
                if len(path) > len(best):
                    best = list(path)
            nbrs = [v for v in adj[u] if v not in on]
            if not nbrs:
                break
            v = rng.choice(nbrs)
            on.add(v)
            path.append(v)
            u = v
    return best

def _solve_one(edges, deadline):
    n = 0
    for u, v in edges:
        if u + 1 > n:
            n = u + 1
        if v + 1 > n:
            n = v + 1
    # n should equal the instance n; but compute from max vertex +1
    # Actually instance uses range(n) so max vertex is n-1.
    if n == 0:
        return []
    adj_full = [set() for _ in range(n)]
    for u, v in edges:
        adj_full[u].add(v)
        adj_full[v].add(u)

    remaining = set(edges)
    cycles = []
    rng = random.Random(0)

    # Greedy extraction
    while remaining:
        if time.time() > deadline:
            break
        adj = [set() for _ in range(n)]
        for u, v in remaining:
            adj[u].add(v)
            adj[v].add(u)
        c = _rand_long_cycle(adj, n, rng, 300, deadline)
        if len(c) < 3:
            break
        ces = set(_cycle_edges(c))
        if not ces <= remaining:
            # shouldn't happen
            break
        cycles.append(c)
        remaining -= ces

    # Local search
    def savings():
        return sum(len(c) - 1 for c in cycles)

    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Move 1: extend/replace each cycle
        for i, c in enumerate(list(cycles)):
            if time.time() > deadline:
                break
            ces = set(_cycle_edges(c))
            # available edges = ces | remaining
            avail = ces | remaining
            adj = [set() for _ in range(n)]
            for u, v in avail:
                adj[u].add(v)
                adj[v].add(u)
            nc = _rand_long_cycle(adj, n, rng, 200, deadline)
            if len(nc) >= 3:
                nces = set(_cycle_edges(nc))
                if nces <= avail and len(nc) > len(c):
                    # replace: freed edges = ces - nces go to remaining
                    freed = ces - nces
                    used = nces - ces
                    cycles[i] = nc
                    remaining = (remaining - used) | freed
                    improved = True
        # Move 2: add a new cycle from remaining
        if time.time() < deadline and remaining:
            adj = [set() for _ in range(n)]
            for u, v in remaining:
                adj[u].add(v)
                adj[v].add(u)
            nc = _rand_long_cycle(adj, n, rng, 200, deadline)
            if len(nc) >= 3:
                nces = set(_cycle_edges(nc))
                if nces <= remaining:
                    cycles.append(nc)
                    remaining -= nces
                    improved = True
        # Move 3: try to remove a cycle and re-decompose its edges into better
        # (skip for simplicity)
    # Build pieces
    pieces = [list(c) for c in cycles]
    for e in remaining:
        pieces.append([e[0], e[1]])
    return pieces

def construct():
    t0 = time.time()
    deadline = t0 + 120  # safety margin before 150s kill
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    result = []
    for spec in specs:
        edges = _instance(*spec)
        pieces = _solve_one(edges, deadline)
        result.append(pieces)
    return result
