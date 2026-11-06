"""Rewritten: exact maximum-weight independent set on the lcm-conflict graph."""
# EVOLVE-BLOCK-START
import time
import math
from fractions import Fraction

def _solve(m, n, max_weight):
    """Maximum-weight independent set on the graph with adjacency lists `m`.
    Nodes are 0..n-1. Returns (best_weight, best_mask)."""
    if n == 0:
        return 0, 0
    adj = [m[i] for i in range(n)]
    W = [max_weight[i] for i in range(n)]
    best = [0]
    best_mask = [0]
    P = [1 << i for i in range(n)]
    Pset = (1 << n) - 1
    INF = 10 ** 30

    def dfs(S, weight, mask):
        if weight + sum(W[i] for i in range(n) if (S >> i) & 1) <= best[0]:
            return
        if S == 0:
            if weight > best[0]:
                best[0] = weight
                best_mask[0] = mask
            return
        # choose pivot with max degree inside S
        pivot = -1
        pdeg = -1
        for i in range(n):
            if (S >> i) & 1:
                d = (adj[i] & S).bit_count()
                if d > pdeg:
                    pdeg = d
                    pivot = i
        if pivot >= 0:
            cand = S & ~adj[pivot]
        else:
            cand = S
        # fractional upper bound via descending weights
        order = [i for i in range(n) if (cand >> i) & 1]
        order.sort(key=lambda i: -W[i])
        frac = weight
        for i in order:
            frac += W[i]
            if frac > best[0]:
                break
        if frac <= best[0]:
            return
        # branch: exclude pivot, then include each candidate
        dfs(S & ~P[pivot], weight, mask)
        for i in order:
            dfs(cand & (cand - 1) & ~adj[i], weight + W[i], mask | P[i])

    dfs(Pset, 0, 0)
    return best[0], best_mask[0]


def _local_improve(S, n, m, max_weight):
    """Greedy add / drop-swap local search to improve an independent set."""
    changed = True
    while changed:
        changed = False
        # try adding
        for i in range(n):
            if not (S >> i) & 1:
                if (m[i] & S) == 0:
                    S |= 1 << i
                    changed = True
        # try drop-swap: drop j, add i if net weight improves
        for j in range(n):
            if (S >> j) & 1:
                S1 = S & ~(1 << j)
                for i in range(n):
                    if not (S1 >> i) & 1 and (m[i] & S1) == 0:
                        if max_weight[i] > max_weight[j]:
                            S = S1 | (1 << i)
                            changed = True
                            break
    return S


def construct():
    t0 = time.time()
    DEADLINE = t0 + 50.0  # safety margin under 90s

    N = 256
    L = [1] * (N + 1)
    for i in range(2, N + 1):
        for j in range(2, N + 1):
            L[i * j] = max(L[i * j], i)

    # conflict graph: edge where lcm(a,b) <= 256
    m = [0] * (N - 1)
    for a in range(2, N + 1):
        ai = a - 2
        for b in range(a + 1, N + 1):
            if a * b // math.gcd(a, b) <= N:
                bi = b - 2
                m[ai] |= 1 << bi
                m[bi] |= 1 << ai

    n = N - 1
    max_weight = [Fraction(1, a) for a in range(2, N + 1)]

    # fast greedy seed (descending weight)
    S = 0
    for a in range(N, 1, -1):
        i = a - 2
        if not (S >> i) & 1 and (m[i] & S) == 0:
            S |= 1 << i
    S = _local_improve(S, n, m, max_weight)
    seed_weight = sum(max_weight[i] for i in range(n) if (S >> i) & 1)
    best = [S, seed_weight]

    # exact maximum-weight independent set (time-bounded)
    if time.time() < DEADLINE:
        try:
            w, mask = _solve(m, n, max_weight)
            if w > best[1]:
                S = _local_improve(mask, n, m, max_weight)
                best = [S, sum(max_weight[i] for i in range(n) if (S >> i) & 1)]
        except Exception:
            pass

    out = [i + 2 for i in range(n) if (best[0] >> i) & 1]
    return out
# EVOLVE-BLOCK-END
