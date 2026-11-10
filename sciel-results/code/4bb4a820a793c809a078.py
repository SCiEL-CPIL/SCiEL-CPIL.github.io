"""Cycle decomposition of Erdős-Gallai-style random graphs: minimize piece count."""
import time

_DEADLINE = 120.0
_START = time.time()


def _remaining():
    return _DEADLINE - time.time()


def _instance(n, p, seed):
    from random import Random
    rng = Random(seed)
    order = list(range(n))
    rng.shuffle(order)
    edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
    edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
    return sorted(edges)


def _build_adj(edges, n):
    adj = [set() for _ in range(n)]
    for (u, v) in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def _dfs_cycle(adj, start, target, max_depth, used):
    """Return a cycle path [start..target] closing to start, or None.
    target is the vertex that must close the cycle (adjacent to start)."""
    if max_depth < 3:
        return None
    path = [start]
    path_set = {start}
    # neighbors to explore, ordered to prefer target when depth is tight
    stack = [(start, iter(sorted(adj[start])))]
    # iterative DFS
    # We'll do recursive-style with explicit stack for backtracking order
    def dfs(v, depth):
        if depth == max_depth - 1:
            if target in adj[v] and target not in path_set:
                return [v]
            return None
        for w in adj[v]:
            if w == start:
                continue
            if w in path_set:
                continue
            if w not in used:
                continue
            path_set.add(w)
            path.append(w)
            res = dfs(w, depth + 1)
            if res is not None:
                return res
            path.pop()
            path_set.discard(w)
        return None
    res = dfs(start, 1)
    if res is None:
        return None
    # res is [..v] where v adjacent to target
    return path + [target]


def _longest_cycle(adj, used, budget, start_hint=None):
    """Find a reasonably long cycle using only edges in `used` (active edge set).
    Returns list of vertices (cycle, closing edge implied) or None."""
    # candidate starting vertices: those with >=2 active neighbors
    cands = [v for v in range(len(adj)) if len(adj[v] & used) >= 2]
    if not cands:
        return None
    if start_hint is not None and len(adj[start_hint] & used) >= 2:
        cands = [start_hint] + [c for c in cands if c != start_hint]
    best = None
    for s in cands:
        if budget <= 0:
            break
        # For each neighbor pair, try to close a cycle through s.
        neigh = sorted(adj[s] & used)
        # try target = each neighbor; find path from s to target
        # To keep it bounded, try increasing max_depth
        found = None
        for max_depth in range(3, 13):
            for t in neigh:
                if t == s:
                    continue
                # quick prune: need a path s->...->t of length max_depth-1 with internal distinct
                cyc = _dfs_cycle(adj, s, t, max_depth, used)
                if cyc is not None:
                    found = cyc
                    break
            if found is not None:
                break
        if found is not None:
            if best is None or len(found) > len(best):
                best = found
    return best


def _greedy_decompose(edges, n, budget=20.0):
    """Greedy: repeatedly extract a long cycle from remaining edges."""
    remaining = set(tuple(e) for e in edges)
    pieces = []
    while remaining:
        if time.time() - _START > budget:
            break
        adj = _build_adj(remaining, n)
        # try a few random start hints to diversify
        cyc = _longest_cycle(adj, remaining, budget)
        if cyc is None:
            # no cycle: take an edge
            e = next(iter(remaining))
            pieces.append([e[0], e[1]])
            remaining.discard(e)
            continue
        # remove cycle edges
        L = len(cyc)
        for i in range(L):
            a = cyc[i]
            b = cyc[(i + 1) % L]
            remaining.discard(tuple(sorted((a, b))))
        pieces.append(cyc)
    # remaining edges as single edges
    for e in remaining:
        pieces.append([e[0], e[1]])
    return pieces


def _improve_by_edge_order(edges, n, budget=30.0):
    """Try a few different greedy orderings by perturbing cycle start selection."""
    base = _greedy_decompose(edges, n, budget)
    best = base
    best_len = len(base)
    # Try removing edges in different orders isn't easy; instead try multiple
    # random restarts with shuffled neighbor exploration.
    import random
    rng = random.Random(12345)
    attempts = 0
    while time.time() - _START < budget and attempts < 25:
        attempts += 1
        # randomized greedy: at each step pick a random start vertex
        remaining = set(tuple(e) for e in edges)
        pieces = []
        while remaining and time.time() - _START < budget:
            adj = _build_adj(remaining, n)
            cands = [v for v in range(n) if len(adj[v] & remaining) >= 2]
            if not cands:
                break
            s = rng.choice(cands)
            cyc = _longest_cycle(adj, remaining, budget, start_hint=s)
            if cyc is None:
                e = next(iter(remaining))
                pieces.append([e[0], e[1]])
                remaining.discard(e)
                continue
            L = len(cyc)
            for i in range(L):
                a = cyc[i]
                b = cyc[(i + 1) % L]
                remaining.discard(tuple(sorted((a, b))))
            pieces.append(cyc)
        for e in remaining:
            pieces.append([e[0], e[1]])
        if len(pieces) < best_len:
            best = pieces
            best_len = len(pieces)
    return best


def _decompose_one(edges, n, budget):
    # quick baseline
    base = _greedy_decompose(edges, n, min(10.0, budget))
    if time.time() - _START > budget - 5:
        return base
    return _improve_by_edge_order(edges, n, budget)


def construct():
    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    results = []
    # distribute budget, leaving safety margin
    total_budget = _DEADLINE - 10.0
    per = total_budget / len(specs)
    for (n, p, seed) in specs:
        edges = _instance(n, p, seed)
        if time.time() - _START > _DEADLINE - 15:
            # not enough time: fallback to all edges
            results.append([list(e) for e in edges])
            continue
        budget = min(per, _DEADLINE - time.time() - 15)
        pieces = _decompose_one(edges, n, max(budget, 2.0))
        results.append(pieces)
    return results
