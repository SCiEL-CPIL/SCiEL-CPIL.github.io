"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

LIMIT = 512
N = LIMIT

def _lcm(a, b):
    return a // gcd(a, b) * b

def _valid_subset(s):
    s = list(s)
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            if _lcm(s[i], s[j]) > LIMIT:
                return False
    return len(s) > 0

def _divisors(m):
    res = []
    for d in range(1, m + 1):
        if m % d == 0:
            res.append(d)
    return res

def _best_divisor_set():
    # find m in [1, LIMIT] with the most divisors; return its divisor set (all <= LIMIT, valid)
    best = None
    best_len = 0
    for m in range(1, LIMIT + 1):
        dv = _divisors(m)
        if len(dv) > best_len:
            best_len = len(dv)
            best = dv
    return best

def _greedy_independent(conflict, order):
    chosen = []
    for v in order:
        ok = True
        for u in chosen:
            if conflict[v][u]:
                ok = False
                break
        if ok:
            chosen.append(v)
    return chosen

def _branch_and_bound(conflict, deadline, init_size):
    # Tomita-style branch and bound for maximum independent set
    # Nodes 0..N-1 represent integers 1..N
    import sys
    sys.setrecursionlimit(200000)

    adj = [0] * N
    for i in range(N):
        mask = 0
        row = conflict[i]
        for j in range(N):
            if i != j and row[j]:
                mask |= (1 << j)
        adj[i] = mask

    def popcount(x):
        return bin(x).count('1')

    def bits_to_list(x):
        out = []
        i = 0
        while x:
            if x & 1:
                out.append(i)
            x >>= 1
            i += 1
        return out

    def color_bound_bitset(P):
        colors = 0
        remaining = P
        while remaining:
            verts = bits_to_list(remaining)
            deg = {}
            for v in verts:
                deg[v] = popcount(adj[v] & remaining)
            verts.sort(key=lambda v: -deg[v])
            color_class = 0
            new_remaining = remaining
            for v in verts:
                if not (new_remaining & (1 << v)):
                    continue
                if (adj[v] & color_class) == 0:
                    color_class |= (1 << v)
                    new_remaining &= ~(1 << v)
            colors += 1
            remaining = new_remaining
        return colors

    full = (1 << N) - 1

    def greedy_is(P):
        chosen = 0
        remaining = P
        while remaining:
            verts = bits_to_list(remaining)
            deg = {}
            for v in verts:
                deg[v] = popcount(adj[v] & remaining)
            verts.sort(key=lambda v: -deg[v])
            v = verts[0]
            chosen |= (1 << v)
            remaining &= ~(adj[v] | (1 << v))
        return chosen

    best = [init_size]
    best_set = [None]

    init_chosen = greedy_is(full)
    init_count = popcount(init_chosen)
    if init_count > best[0]:
        best[0] = init_count
        best_set[0] = bits_to_list(init_chosen)

    def search(P, current):
        if time.time() > deadline:
            return
        if P == 0:
            c = popcount(current)
            if c > best[0]:
                best[0] = c
                best_set[0] = bits_to_list(current)
            return
        ub = color_bound_bitset(P)
        if popcount(current) + ub <= best[0]:
            return
        verts = bits_to_list(P)
        deg = {}
        for v in verts:
            deg[v] = popcount(adj[v] & P)
        v = max(verts, key=lambda x: deg[x])
        # include v
        search(P & ~(adj[v] | (1 << v)), current | (1 << v))
        # exclude v
        search(P & ~(1 << v), current)

    search(full, 0)

    if best_set[0] is None:
        return list(range(1, init_size + 1))
    return [x + 1 for x in best_set[0]]

def construct():
    t0 = time.time()
    deadline = t0 + 75.0

    # Build conflict graph
    conflict = [[False] * N for _ in range(N)]
    for i in range(N):
        a = i + 1
        for j in range(i + 1, N):
            b = j + 1
            if _lcm(a, b) > LIMIT:
                conflict[i][j] = True
                conflict[j][i] = True

    # Guaranteed valid initial lower bound: divisor set of m with max divisors
    init_set = _best_divisor_set()
    if not _valid_subset(init_set):
        # should not happen, but fallback
        deg = [sum(conflict[i]) for i in range(N)]
        order = sorted(range(N), key=lambda i: deg[i])
        init_set = _greedy_independent(conflict, order)

    init_size = len(init_set)

    if time.time() > deadline:
        return init_set

    result = _branch_and_bound(conflict, deadline, init_size)

    if not _valid_subset(result):
        return init_set
    return result
# EVOLVE-BLOCK-END
