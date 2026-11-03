import time
import random
import math
from collections import deque

# EVOLVE-BLOCK-START

_DEADLINE = 150.0
_MARGIN = 2.0
_START = time.time()


def _time_left():
    return _DEADLINE - _MARGIN - (time.time() - _START)


def _moore(k, g):
    if k < 2:
        return 1
    r = k - 1
    if g % 2 == 1:
        return 1 + k * (r ** ((g - 1) // 2))
    else:
        return 2 * r ** (g // 2)


def _softmin(a, b, t):
    if t <= 0:
        return a if a <= b else b
    m = a if a <= b else b
    e = math.exp(-(a - m) / t)
    f = math.exp(-(b - m) / t)
    return m - t * math.log(e + f)


def _greedy(k, seed, budget):
    rng = random.Random(seed)
    adj = [[]]
    pending = [0]
    loads = [0]
    t = 0.4
    target = 2 * _moore(k, g) if False else None  # not used
    target = None

    def add_vertex():
        idx = len(adj)
        adj.append([])
        loads.append(0)
        return idx

    def link(a, b):
        adj[a].append(b)
        adj[b].append(a)
        loads[a] += 1
        loads[b] += 1

    while _time_left() > budget:
        if pending:
            parent = pending.pop()
        else:
            # choose parent via smooth soft-min over loads (spread the growth)
            m = min(loads)
            best = -1.0
            best_i = -1
            for i, L in enumerate(loads):
                s = _softmin(L, m, t)
                if s > best:
                    best = s
                    best_i = i
            if best_i < 0:
                break
            parent = best_i

        # parent already has one edge to this new vertex
        cur = loads[parent]
        if cur >= k:
            # parent saturated; this vertex must attach to a *different* vertex
            # find a vertex with free capacity and connect
            cand = -1
            best_l = -1
            for i, L in enumerate(loads):
                if L < k and L > best_l:
                    best_l = L
                    cand = i
            if cand < 0:
                # everyone saturated -> we are done
                break
            v = add_vertex()
            link(cand, v)
            parent = cand
            cur = loads[cand]
            # v now has 1 edge (to cand); we still need to give v its other edges
            # but v is a "leaf" so far; continue filling v below
            # (v currently has degree 1, needs k-1 more)
            # fall through to fill v's remaining edges with fresh vertices
            need = k - 1
        else:
            v = add_vertex()
            link(parent, v)
            need = k - 1  # v needs k-1 more neighbours (all fresh)

        # give v its remaining neighbours as fresh vertices
        for _ in range(need):
            nv = add_vertex()
            link(v, nv)
            pending.append(nv)

    n = len(adj)
    if n < 2 * k:
        return None
    edges = []
    for i in range(n):
        for j in adj[i]:
            if i < j:
                edges.append((i, j))
    return n, edges


def _is_valid(k, g, n, edges):
    # degree + simple
    deg = [0] * n
    seen = set()
    for i, j in edges:
        if i < 0 or j < 0 or i >= n or j >= n or i == j:
            return False
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in seen:
            return False
        seen.add((a, b))
        deg[a] += 1
        deg[b] += 1
    for d in deg:
        if d != k:
            return False
    # build adjacency
    adj = [set() for _ in range(n)]
    for a, b in seen:
        adj[a].add(b)
        adj[b].add(a)
    # exact girth via non-backtracking walks from every vertex
    for s in range(n):
        # state: (v, prev) ; depth = 1 (we are at v, came from prev)
        dist = {}
        q = deque()
        for u in adj[s]:
            st = (u, s)
            dist[st] = 1
            q.append(st)
        while q:
            v, prev = q.popleft()
            d = dist[(v, prev)]
            if d + 1 >= g:
                continue
            for u in adj[v]:
                if u == prev:
                    continue
                st = (u, v)
                if st in dist:
                    # closed a non-backtracking cycle of length d + 1
                    if d + 1 < g:
                        return False
                else:
                    dist[st] = d + 1
                    q.append(st)
    return True


def construct(k, g):
    global _START
    _START = time.time()
    deadline = _DEADLINE - _MARGIN
    best = None
    best_n = None
    seed = 0
    while time.time() - _START < deadline:
        # per-attempt budget shrinks as we approach the deadline
        rem = deadline - (time.time() - _START)
        budget = min(6.0, max(0.4, rem * 0.25))
        if rem < 0.6:
            break
        res = _greedy(k, seed, budget)
        seed += 1
        if res is None:
            continue
        n, edges = res
        # cheap pre-checks before the expensive girth test
        deg = [0] * n
        for i, j in edges:
            deg[i] += 1
            deg[j] += 1
        if any(d != k for d in deg):
            continue
        if not _is_valid(k, g, n, edges):
            continue
        if best_n is None or n < best_n:
            best_n = n
            best = (n, edges)
            # if we hit the Moore bound, it's optimal
            if n <= _moore(k, g):
                break
    if best is None:
        # last resort: one more small deterministic attempt with tight budget
        res = _greedy(k, 12345, 0.5)
        if res is not None:
            n, edges = res
            deg = [0] * n
            for i, j in edges:
                deg[i] += 1
                deg[j] += 1
            if all(d == k for d in deg) and _is_valid(k, g, n, edges):
                best = (n, edges)
    if best is None:
        # absolute fallback: return the smallest valid complete-ish graph we can
        # (should not happen); return a valid k-regular graph if k allows
        raise RuntimeError("no valid graph found within budget")
    return best
# EVOLVE-BLOCK-END
