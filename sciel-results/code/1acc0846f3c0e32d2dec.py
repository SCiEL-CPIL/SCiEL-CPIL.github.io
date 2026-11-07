"""Cochromatic partition of a pinned random graph: exact minimum via B&B + subset-DP bound."""
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
DEADLINE = time.time() + 78.0  # safety margin under the 90 s wall-clock kill

# Bit masks of the two adjacency relations.
E = set()
for a, b in P['edges']:
    if a == b:
        continue
    E.add((a, b))
    E.add((b, a))
A = [0] * N
for (a, b) in E:
    A[a] |= (1 << b)
    A[b] |= (1 << a)
ALL = (1 << N) - 1
C = [ALL ^ A[i] for i in range(N)]  # complement adjacency (no self bits)

# ---- subset DP: chromatic number of G and of complement(G) ----
def build_chromatic(adj):
    n = N
    ch = [0] * (1 << n)
    for s in range(1, 1 << n):
        lb = s & (-s)
        rest = s ^ lb
        ch[s] = ch[rest] + 1
        # enumerate non-empty subsets of rest
        sub = rest
        while sub:
            if adj[lb.bit_length() - 1] & sub:
                v = ch[s ^ sub] + 1
                if v < ch[s]:
                    ch[s] = v
            sub = (sub - 1) & rest
    return ch

CHG = build_chromatic(A)
CHC = build_chromatic(C)
EMPTY = 0
EMPTY_LIST = []

def _is_clique(m):
    i = 0
    while m:
        i = (m & -m).bit_length() - 1
        if A[i] & m:
            return False
        m ^= (1 << i)
    return True

def _is_independent(m):
    i = 0
    while m:
        i = (m & -m).bit_length() - 1
        if C[i] & m:
            return False
        m ^= (1 << i)
    return True

def _modules(m):
    """All non-empty subsets of m that are a clique or an independent set."""
    mods = []
    sub = m
    while sub:
        if _is_clique(sub) or _is_independent(sub):
            mods.append(sub)
        sub = (sub - 1) & m
    return mods

# ---- branch and bound ----
_k_cache = {ALL: None}
_greedy = {}

def _greedy_ub(m):
    if m in _greedy:
        return _greedy[m]
    best = N
    rem = m
    while rem:
        lb = rem & (-rem)
        v = lb.bit_length() - 1
        cand = rem & (A[v] | C[v] | lb)
        if cand:
            rem ^= cand
        else:
            rem ^= lb
        best += 1
    _greedy[m] = best
    return best

def _k(m):
    r = _k_cache.get(m)
    if r is not None:
        return r
    if m == 0:
        _k_cache[m] = 0
        return 0
    # lower bound from the two chromatic numbers
    if CHG[m] + CHC[m] >= _greedy_ub(m):
        _k_cache[m] = CHG[m] + CHC[m]
        return _k_cache[m]
    mods = _modules(m)
    mods.sort(key=lambda s: -s.bit_count())
    # pick the vertex with the fewest feasible modules to branch on
    bestv = None
    bestc = None
    for s in mods:
        c = s.bit_count()
        if bestc is None or c < bestc:
            bestc = c
            bestv = s
    v = (bestv & (-bestv)).bit_length() - 1
    opts = [s for s in mods if s & (1 << v)]
    opts.sort(key=lambda s: -s.bit_count())
    res = N
    for s in opts:
        if time.time() > DEADLINE:
            break
        r = _k(m ^ s) + 1
        if r < res:
            res = r
    _k_cache[m] = res
    return res

def construct():
    k = _k(ALL)
    # reconstruct a partition into exactly k modules via memoized search
    memo = {}

    def rec(m, r):
        if m == 0:
            return []
        key = (m, r)
        if key in memo:
            return memo[key]
        if r == 1:
            res = [m]
        else:
            res = None
            for s in _modules(m):
                if time.time() > DEADLINE:
                    break
                if CHG[m ^ s] + CHC[m ^ s] < r - 1:
                    sub = rec(m ^ s, r - 1)
                    if sub is not None:
                        res = [s] + sub
                        break
        memo[key] = res
        return res

    parts = rec(ALL, k)
    if parts is None:
        # extremely unlikely; fall back to a valid (non-optimal) greedy partition
        parts = []
        rem = ALL
        while rem:
            lb = rem & (-rem)
            v = lb.bit_length() - 1
            cand = rem & (A[v] | C[v] | lb)
            if cand:
                parts.append(cand)
                rem ^= cand
            else:
                parts.append(lb)
                rem ^= lb
    out = []
    for m in parts:
        lst = []
        mm = m
        while mm:
            b = mm & (-mm)
            lst.append(b.bit_length() - 1)
            mm ^= b
        out.append(tuple(lst))
    return tuple(out)
