"""Suite task: dense odd-order graph avoiding equal-degree path endpoints.

Search design (kept in the same family as the seed, improved):
  * Fast O(n^3) validity check via a precomputed per-edge "forbidden" set.
  * Greedy construction that only adds an edge when it keeps the graph valid.
  * Local-search repair: repeatedly try to add the best currently-forbidden
    edge (the one that removes the fewest other edges), then re-validate.
  * Explicit wall-clock deadline guard returns the best valid graph early.
"""
import time
import random

# EVOLVE-BLOCK-START
_N = 15
_ALL_EDGES = None
_DEADLINE = 90.0
_START = None

def _init():
    global _ALL_EDGES, _START, _DEADLINE
    if _ALL_EDGES is None:
        _ALL_EDGES = [(u, v) for u in range(_N) for v in range(u + 1, _N)]
    _START = time.time()
    _DEADLINE = _START + 90.0

def _deadline():
    return time.time() >= _DEADLINE

def _is_valid(adj, deg):
    """Return True iff no simple 3-edge path s-u-v-t has deg(s)==deg(t)."""
    for u in range(_N):
        for v in adj[u]:
            if v <= u:
                continue
            forbid = set()
            for s in adj[u]:
                if s != v:
                    forbid.add(deg[s])
            for t in adj[v]:
                if t != u:
                    if deg[t] in forbid:
                        return False
    return True

def _greedy_build():
    """Add edges in a fixed order, keeping the graph valid at every step."""
    adj = [set() for _ in range(_N)]
    deg = [0] * _N
    count = 0
    for u, v in _ALL_EDGES:
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        if _is_valid(adj, deg):
            count += 1
        else:
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
    return adj, deg, count

def _local_search(adj, deg, count):
    """Try to add the best forbidden edge, re-validate, keep if it improves."""
    adj = [set(x) for x in adj]
    deg = list(deg)
    count = int(count)
    while not _deadline():
        best_e = None
        best_rem = None
        best_cost = None
        for u, v in _ALL_EDGES:
            if v in adj[u]:
                continue
            forbid = set()
            for s in adj[u]:
                if s != v:
                    forbid.add(deg[s])
            rem = 0
            for t in adj[v]:
                if t != u and deg[t] in forbid:
                    rem += 1
            cost = rem - 1
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_e = (u, v)
                best_rem = rem
        if best_e is None:
            break
        if best_cost >= 0:
            break
        u, v = best_e
        adj[u].add(v)
        adj[v].add(u)
        deg[u] += 1
        deg[v] += 1
        if _is_valid(adj, deg):
            count += 1
        else:
            adj[u].remove(v)
            adj[v].remove(u)
            deg[u] -= 1
            deg[v] -= 1
    return adj, deg, count

def _to_edges(adj):
    edges = []
    for u in range(_N):
        for v in adj[u]:
            if v > u:
                edges.append([int(u), int(v)])
    return edges

def construct():
    _init()
    best_adj = None
    best_deg = None
    best_count = -1

    # Phase 1: greedy build.
    adj, deg, count = _greedy_build()
    if count > best_count:
        best_adj, best_deg, best_count = adj, deg, count

    # Phase 2: local-search repair on the greedy result.
    if not _deadline():
        adj, deg, count = _local_search(best_adj, best_deg, best_count)
        if count > best_count:
            best_adj, best_deg, best_count = adj, deg, count

    # Phase 3: a few randomized restarts if time remains.
    rng = random.Random(12345)
    while not _deadline():
        adj = [set() for _ in range(_N)]
        deg = [0] * _N
        count = 0
        order = list(_ALL_EDGES)
        rng.shuffle(order)
        for u, v in order:
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
            if _is_valid(adj, deg):
                count += 1
            else:
                adj[u].remove(v)
                adj[v].remove(u)
                deg[u] -= 1
                deg[v] -= 1
        if count > best_count:
            best_adj, best_deg, best_count = adj, deg, count
        if not _deadline():
            adj, deg, count = _local_search(adj, deg, count)
            if count > best_count:
                best_adj, best_deg, best_count = adj, deg, count

    if best_adj is None:
        return []
    return _to_edges(best_adj)
# EVOLVE-BLOCK-END
