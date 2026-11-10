"""Rewritten program: greedy longest-cycle extraction for edge decomposition."""
# EVOLVE-BLOCK-START
import time
from random import Random


def _build_adjacency(edges):
    adj = {}
    for (u, v) in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _find_long_cycle(adj, start, rng, max_iter=200000):
    """Randomized DFS to find a long simple cycle through start.
    Returns list of vertices forming the cycle (length >= 3) or None.
    adj is a dict of vertex -> set of neighbors (mutable remaining graph)."""
    best = None
    best_len = 2
    for _ in range(40):  # several randomized attempts
        # greedy randomized walk with backtracking
        path = [start]
        on_path = {start}
        # We'll do a depth-first search trying to close a cycle back to start
        # with a greedy preference for unvisited neighbors.
        stack = [(start, iter(sorted(adj.get(start, ()), key=lambda x: rng.random())))]
        visited = {start}
        found = None
        iters = 0
        while stack and iters < max_iter:
            iters += 1
            node, it = stack[-1]
            advanced = False
            for nb in it:
                if nb == start:
                    # closing a cycle
                    cur_len = len(path)
                    if cur_len >= 3 and cur_len > best_len:
                        best_len = cur_len
                        best = path[:]
                    # continue searching for even longer? we'll keep going
                    advanced = True
                    break
                if nb not in visited:
                    visited.add(nb)
                    path.append(nb)
                    stack.append((nb, iter(sorted(adj.get(nb, ()), key=lambda x: rng.random()))))
                    advanced = True
                    break
            if not advanced:
                # backtrack
                visited.discard(node)
                stack.pop()
                if path and path[-1] == node:
                    path.pop()
        if found is None and best is not None:
            pass
        # also try to just greedily extend a path and close
        if best is None:
            # simple greedy cycle: walk until we hit start
            cur = start
            trail = [cur]
            seen = {cur}
            for _ in range(len(adj) + 5):
                nbrs = [x for x in adj.get(cur, ()) if x not in seen]
                if not nbrs:
                    break
                nxt = rng.choice(nbrs)
                if nxt == start and len(trail) >= 3:
                    best = trail[:]
                    break
                seen.add(nxt)
                trail.append(nxt)
                cur = nxt
    return best


def _decompose_once(n, edges, rng, time_budget):
    """One attempt at decomposition. Returns (pieces, is_valid)."""
    start_t = time.time()
    # remaining edges
    rem = set()
    for e in edges:
        rem.add(tuple(sorted(e)))
    adj = {}
    for (u, v) in rem:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    pieces = []
    # vertices list
    verts = list(range(n))

    while rem:
        if time.time() - start_t > time_budget:
            # bail: emit remaining as single edges
            for (u, v) in rem:
                pieces.append([u, v])
            return pieces
        # pick a start vertex with highest degree among remaining
        # to bias toward dense areas
        degs = {v: len(adj.get(v, ())) for v in adj}
        if not degs:
            break
        # choose start randomly among vertices with degree >= 2
        cands = [v for v, d in degs.items() if d >= 2]
        if not cands:
            break
        start = rng.choice(cands)
        cycle = _find_long_cycle(adj, start, rng, max_iter=100000)
        if cycle is None or len(cycle) < 3:
            # no cycle found; emit one edge as single piece to make progress
            e = next(iter(rem))
            u, v = e
            rem.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
            continue
        # extract cycle edges
        cyc = cycle
        # ensure distinct
        if len(set(cyc)) < len(cyc) or len(cyc) < 3:
            e = next(iter(rem))
            u, v = e
            rem.discard(e)
            adj[u].discard(v)
            adj[v].discard(u)
            pieces.append([u, v])
            continue
        for i in range(len(cyc)):
            a = cyc[i]
            b = cyc[(i + 1) % len(cyc)]
            e = tuple(sorted((a, b)))
            if e not in rem:
                # should not happen; fallback
                break
            rem.discard(e)
            adj[a].discard(b)
            adj[b].discard(a)
        else:
            pieces.append(cyc)
            continue
        # if break happened, emit single edge
        e = next(iter(rem))
        u, v = e
        rem.discard(e)
        adj[u].discard(v)
        adj[v].discard(u)
        pieces.append([u, v])

    # emit any remaining
    for (u, v) in rem:
        pieces.append([u, v])
    return pieces


def _validate(n, edges, pieces):
    """Check pieces are edge-disjoint and cover all edges, cycles valid."""
    edge_set = set(tuple(sorted(e)) for e in edges)
    used = set()
    for p in pieces:
        if len(p) == 2:
            e = tuple(sorted(p))
            if e in used:
                return False
            if e not in edge_set:
                return False
            used.add(e)
        else:
            if len(p) < 3 or len(set(p)) != len(p):
                return False
            for i in range(len(p)):
                a = p[i]
                b = p[(i + 1) % len(p)]
                e = tuple(sorted((a, b)))
                if e in used:
                    return False
                if e not in edge_set:
                    return False
                used.add(e)
    return used == edge_set


def construct():
    from random import Random

    def instance(n, p, seed):
        rng = Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = {tuple(sorted(e)) for e in zip(order, order[1:])}
        edges |= {(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p}
        return sorted(edges)

    specs = ((20, 0.4, 11), (25, 0.35, 12), (30, 0.3, 13))
    total_budget = 100.0  # seconds
    results = []
    t0 = time.time()
    for (n, p, seed) in specs:
        edges = instance(n, p, seed)
        # try several seeds for the decomposition RNG, keep best
        best_pieces = None
        best_count = None
        per_graph_budget = (total_budget - (time.time() - t0)) / (len(specs) - specs.index((n, p, seed)))
        per_graph_budget = max(5.0, min(35.0, per_graph_budget))
        for attempt in range(6):
            if time.time() - t0 > total_budget - 5.0:
                break
            rng = Random(seed * 1000003 + attempt * 7919 + 1)
            pieces = _decompose_once(n, edges, rng, time_budget=per_graph_budget / 6.0)
            if not _validate(n, edges, pieces):
                continue
            cnt = len(pieces)
            if best_count is None or cnt < best_count:
                best_count = cnt
                best_pieces = pieces
        if best_pieces is None:
            # fallback: single edges
            best_pieces = [list(e) for e in edges]
        results.append(best_pieces)
    return results
# EVOLVE-BLOCK-END
