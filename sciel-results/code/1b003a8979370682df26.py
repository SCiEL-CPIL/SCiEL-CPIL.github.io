import time
import random

_DEADLINE = 144.0
_START = None

def _elapsed():
    return time.time() - _START

def _is_c4_free(adj):
    n = len(adj)
    for i in range(n):
        ni = adj[i]
        x = ni
        while x:
            lsb = x & (-x)
            j = lsb.bit_length() - 1
            x ^= lsb
            common = ni & adj[j]
            if common & (common - 1):
                return False
    return True

def _find_independent_set(adj, k):
    # returns list of vertices forming independent set of size k, or None
    n = len(adj)
    deg = [adj[i].bit_count() for i in range(n)]
    order = sorted(range(n), key=lambda i: -deg[i])
    result = []
    import sys
    sys.setrecursionlimit(1000000)

    def bt(cand, size):
        if size >= k:
            return True
        if cand.bit_count() < k - size:
            return False
        lsb = cand & (-cand)
        v = lsb.bit_length() - 1
        result.append(v)
        if bt(cand & ~adj[v] & ~lsb, size + 1):
            return True
        result.pop()
        if bt(cand & ~lsb, size):
            return True
        return False

    allcand = (1 << n) - 1
    if bt(allcand, 0):
        return result[:k]
    return None

def _repair_alpha(adj, n, rng, max_rounds=2000):
    for _ in range(max_rounds):
        if _elapsed() > _DEADLINE:
            break
        S = _find_independent_set(adj, 11)
        if S is None:
            break
        # try to add an edge within S with 0 common neighbors
        added = False
        pairs = list(zip(S, S))
        rng.shuffle(pairs)
        for (i, j) in pairs:
            if i >= j:
                continue
            if (adj[i] >> j) & 1:
                continue
            if (adj[i] & adj[j]) == 0:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                added = True
                break
        if not added:
            # no safe edge within S; try random safe edge anywhere
            for _ in range(50):
                i = rng.randrange(n); j = rng.randrange(n)
                if i != j and not ((adj[i] >> j) & 1) and (adj[i] & adj[j]) == 0:
                    adj[i] |= (1 << j); adj[j] |= (1 << i)
                    added = True
                    break
        if not added:
            break
    return adj

def _greedy_c4_free(n, rng):
    adj = [0] * n
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    rng.shuffle(pairs)
    for (i, j) in pairs:
        if (adj[i] & adj[j]) == 0:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
    return adj

def _edges_from_adj(adj, n):
    edges = []
    for i in range(n):
        x = adj[i]
        while x:
            lsb = x & (-x)
            j = lsb.bit_length() - 1
            x ^= lsb
            if i < j:
                edges.append([i, j])
    return edges

def construct():
    global _START
    _START = time.time()
    rng = random.Random(777)
    best = None
    for n in range(39, 29, -1):
        if _elapsed() > _DEADLINE - 3:
            break
        for attempt in range(30):
            if _elapsed() > _DEADLINE - 3:
                break
            adj = _greedy_c4_free(n, rng)
            adj = _repair_alpha(adj, n, rng)
            if _is_c4_free(adj) and _find_independent_set(adj, 11) is None:
                best = (n, adj)
                break
        if best is not None:
            break
    if best is None:
        adj = [0]*30
        for t in range(10):
            a,b,c = 3*t,3*t+1,3*t+2
            adj[a] |= (1<<b)|(1<<c); adj[b] |= (1<<a)|(1<<c); adj[c] |= (1<<a)|(1<<b)
        return {'n':30,'edges':_edges_from_adj(adj,30)}
    n, adj = best
    return {'n':n,'edges':_edges_from_adj(adj,n)}
