"""Construct a k-regular graph of large girth via random regular graph + 2-switch repair.

Approach family (new, not algebraic incidence): probabilistic construction.
  1. Sample a random k-regular simple graph on a vertex count chosen near the Moore bound.
  2. Certify girth exactly with non-backtracking BFS from every vertex.
  3. While girth < g, try random 2-switches (preserve k-regularity) that do not create
     loops/multiedges and that increase the certified girth; accept improving moves.
  4. Return the best graph found within the time budget.
"""
import random
import time
import math
import sys
from collections import deque

# EVOLVE-BLOCK-START

def _moore_bound(k, g):
    """Moore bound lower bound on n for a k-regular graph of girth g."""
    if g % 2 == 1:
        # odd girth: n >= 1 + k * sum_{i=0}^{(g-3)/2} (k-1)^i
        r = (g - 3) // 2
        s = 0
        for i in range(r + 1):
            s += (k - 1) ** i
        return 1 + k * s
    else:
        # even girth: n >= 2 * sum_{i=0}^{g/2 - 1} (k-1)^i
        r = g // 2 - 1
        s = 0
        for i in range(r + 1):
            s += (k - 1) ** i
        return 2 * s


def _girth_nbt(adj):
    """Exact girth via non-backtracking BFS from every vertex.
    adj: list of lists of neighbors (simple, undirected).
    Returns the length of the shortest cycle (>=3) or a large number if none.
    """
    n = len(adj)
    best = float('inf')
    # state: (vertex, prev) meaning we arrived at vertex from prev (non-backtracking)
    # dist[v][prev] = depth
    for start in range(n):
        # BFS over non-backtracking walks
        dist = {}
        for nb in adj[start]:
            dist[(nb, start)] = 1
        q = deque(dist.keys())
        while q:
            v, prev = q.popleft()
            d = dist[(v, prev)]
            if d >= best:
                continue
            for w in adj[v]:
                if w == prev:
                    continue
                key = (w, v)
                if key in dist:
                    # cycle of length d + 1 (walk from start...v then edge to w which is
                    # already reached at some depth; but for girth we need the walk that
                    # returns to a previously seen state forming a cycle through start)
                    # Standard non-backtracking girth: a cycle is found when we reach
                    # (w, v) where w == start (closed) -> length d+1.
                    if w == start:
                        if d + 1 < best:
                            best = d + 1
                else:
                    dist[key] = d + 1
                    q.append(key)
    return best if best != float('inf') else n * 2


def _random_k_regular(n, k, rng):
    """Configuration model to produce a simple k-regular graph; returns adj or None."""
    # Build half-edges
    stubs = []
    for v in range(n):
        stubs.extend([v] * k)
    # Try to pair into a simple graph
    for _attempt in range(60):
        rng.shuffle(stubs)
        adj = [set() for _ in range(n)]
        ok = True
        for i in range(0, len(stubs), 2):
            a, b = stubs[i], stubs[i + 1]
            if a == b:
                ok = False
                break
            if b in adj[a]:
                ok = False
                break
            adj[a].add(b)
            adj[b].add(a)
        if ok:
            # verify regularity
            if all(len(s) == k for s in adj):
                return [list(s) for s in adj]
    return None


def _try_2switch(adj, k, rng, max_trials=400):
    """Attempt one random 2-switch that keeps graph simple & k-regular.
    Returns new adj (copy) if a valid improving swap is found, else None.
    A 2-switch: pick two edges (a,b),(c,d) with all distinct endpoints, remove them,
    add (a,c),(b,d) or (a,d),(b,c) if both are non-edges and not loops.
    """
    n = len(adj)
    for _ in range(max_trials):
        # pick two random edges
        e1 = (rng.randrange(n), rng.randrange(n))
        a, b = e1
        if b not in adj[a]:
            continue
        e2 = (rng.randrange(n), rng.randrange(n))
        c, d = e2
        if d not in adj[c]:
            continue
        if len({a, b, c, d}) < 4:
            continue
        # candidate new edges
        for (x, y), (u, v) in [((a, c), (b, d)), ((a, d), (b, c))]:
            if x == y or u == v:
                continue
            if y in adj[x] or v in adj[u]:
                continue
            # apply
            new_adj = [list(s) for s in adj]
            new_adj[a].remove(b)
            new_adj[b].remove(a)
            new_adj[c].remove(d)
            new_adj[d].remove(c)
            new_adj[x].append(y)
            new_adj[y].append(x)
            new_adj[u].append(v)
            new_adj[v].append(u)
            return new_adj
    return None


def construct(k, g):
    rng = random.Random(12345)
    t0 = time.time()
    time_budget = 110.0  # leave margin under 120 s

    mb = _moore_bound(k, g)
    # start near Moore bound; round up to a multiple of 2 (for even) just for sanity
    n0 = mb
    # ensure n0 * k is even (handshaking)
    if (n0 * k) % 2 != 0:
        n0 += 1

    best_adj = None
    best_girth = -1

    # Try a few starting sizes (Moore bound and a couple of small bumps) to improve odds
    candidates = []
    for bump in [0, 2, 4, 6, 8, 12, 16]:
        cand = n0 + bump
        if (cand * k) % 2 != 0:
            cand += 1
        if cand not in candidates:
            candidates.append(cand)

    for n in candidates:
        if time.time() - t0 > time_budget:
            break
        # sample several random regular graphs at this size
        for trial in range(8):
            if time.time() - t0 > time_budget:
                break
            adj = _random_k_regular(n, k, rng)
            if adj is None:
                continue
            girth = _girth_nbt(adj)
            if girth > best_girth:
                best_girth = girth
                best_adj = [list(s) for s in adj]
                if girth >= g:
                    return n, _edges_from_adj(adj)

    if best_adj is None:
        # fallback: try a slightly larger size
        n = n0 + 20
        if (n * k) % 2 != 0:
            n += 1
        for trial in range(20):
            adj = _random_k_regular(n, k, rng)
            if adj is not None:
                best_adj = adj
                best_girth = _girth_nbt(adj)
                break
        if best_adj is None:
            raise RuntimeError("could not build a regular graph")

    # Repair loop: increase girth via 2-switches
    adj = best_adj
    cur_girth = best_girth
    stall = 0
    while cur_girth < g and time.time() - t0 < time_budget:
        new_adj = _try_2switch(adj, k, rng, max_trials=60)
        if new_adj is None:
            stall += 1
            if stall > 200:
                break
            continue
        ng = _girth_nbt(new_adj)
        if ng > cur_girth:
            adj = new_adj
            cur_girth = ng
            stall = 0
        else:
            # small chance to accept neutral moves to escape local minima
            if rng.random() < 0.02:
                adj = new_adj
            stall += 1
            if stall > 400:
                break

    n = len(adj)
    return n, _edges_from_adj(adj)


def _edges_from_adj(adj):
    n = len(adj)
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return edges

# EVOLVE-BLOCK-END
