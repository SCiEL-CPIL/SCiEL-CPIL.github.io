"""Decompose three fixed instance graphs into few cycles + edges (Erdős-Gallai task)."""
# EVOLVE-BLOCK-START
import time
import random

_DEADLINE = 150.0
_EPS = 1e-6


def instance(n, p, seed):
    rng = random.Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _cycle_edges(cyc):
    m = len(cyc)
    return {tuple(sorted((cyc[i], cyc[(i + 1) % m]))) for i in range(m)}


def _try_find_cycle(adj, rng, max_iter=400, time_budget=0.15):
    """Greedy longest-path heuristic + local improvement. Returns a vertex cycle or None."""
    n = len(adj)
    start = rng.randrange(n)
    path = [start]
    used = {start}
    while True:
        last = path[-1]
        nxts = [v for v in adj[last] if v not in used]
        if not nxts:
            break
        path.append(rng.choice(nxts))
        used.add(path[-1])
    if len(path) < 3:
        return None
    if path[0] in adj[path[-1]]:
        cyc = path[:]
    else:
        cyc = None

    # Local search: try to close by extending, or shorten from the tail to close.
    t0 = time.time()
    improved = True
    while improved and (time.time() - t0) < time_budget:
        improved = False
        if cyc is None:
            # try extending from tail
            last = path[-1]
            for v in adj[last]:
                if v not in used:
                    cand = path + [v]
                    if cand[0] in adj[cand[-1]]:
                        cyc = cand[:]
                        improved = True
                        break
            if cyc is None:
                # try shortening from tail until it closes
                for i in range(len(path) - 1, 2, -1):
                    if path[0] in adj[path[i - 1]]:
                        cyc = path[:i]
                        improved = True
                        break
            if cyc is None:
                break
        else:
            # try to extend the closed cycle
            last = cyc[-1]
            for v in adj[last]:
                if v not in set(cyc):
                    cand = cyc + [v]
                    if cand[0] in adj[cand[-1]]:
                        cyc = cand
                        improved = True
                        break
    return cyc


def _decompose(edges, rng, deadline):
    n = max((max(e) for e in edges), default=0) + 1
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    remaining = {tuple(sorted(e)) for e in edges}

    pieces = []
    t_start = time.time()
    while remaining:
        if time.time() > deadline:
            break
        found = None
        for _ in range(80):
            if time.time() > deadline:
                break
            cyc = _try_find_cycle(adj, rng, max_iter=300, time_budget=0.05)
            if cyc and len(cyc) > 1:
                ce = _cycle_edges(cyc)
                if ce <= remaining and len(ce) == len(cyc):
                    found = cyc
                    break
        if found is None:
            # no cycle found; emit a single edge
            e = min(remaining)
            pieces.append(list(e))
            remaining.discard(e)
            adj[e[0]].discard(e[1])
            adj[e[1]].discard(e[0])
        else:
            ce = _cycle_edges(found)
            pieces.append(found)
            for eu, ev in ce:
                remaining.discard((eu, ev))
                adj[eu].discard(ev)
                adj[ev].discard(eu)
    # Safety: any leftovers become single edges (valid decomposition).
    for e in sorted(remaining):
        pieces.append(list(e))
    return pieces


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE - 5.0   # safety margin
    rng = random.Random(2024)
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    for i, spec in enumerate(specs):
        edges = instance(*spec)
        if time.time() > deadline:
            # not enough time; fall back to trivial for this graph
            results.append([list(e) for e in edges])
            continue
        pieces = _decompose(edges, rng, deadline)
        results.append(pieces)
    return results
# EVOLVE-BLOCK-END
