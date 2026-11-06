"""Rewritten program: multi-restart greedy + randomized local search for
max-size subset of [1,256] with property: for all a<b<=c in S, a !| (b+c)."""
import time
import random

N = 256
DEADLINE = 85.0

def _can_add(S, x):
    m = len(S)
    # x as smallest (a): x < b <= c, x | (b+c)
    for j in range(m):
        b = S[j]
        if b <= x:
            continue
        for k in range(j, m):
            c = S[k]
            if (b + c) % x == 0:
                return False
    # x as middle (b): a < x <= c, a | (x+c)
    for i in range(m):
        a = S[i]
        if a >= x:
            continue
        for k in range(m):
            c = S[k]
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
    # x as largest (c): a < b <= x, a | (b+x)
    for i in range(m):
        a = S[i]
        if a >= x:
            continue
        for j in range(m):
            b = S[j]
            if b <= a or b > x:
                continue
            if (b + x) % a == 0:
                return False
    return True

def _grow(S, order):
    Sset = set(S)
    improved = True
    while improved:
        improved = False
        for x in order:
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
    S.sort()
    return S

def construct():
    start = time.time()
    rng = random.Random(20240517)
    best = list(range(192, 257))

    def consider(S):
        nonlocal best
        if len(S) > len(best):
            best = list(S)

    def try_start(order, shake=None):
        S = list(best) if shake is None else list(best)
        if shake:
            S = list(set(S) - set(shake))
        S = _grow(S, order)
        consider(S)

    # Phase 1: diverse greedy starts
    base_orders = [list(range(N, 0, -1)), list(range(1, N+1))]
    for _ in range(8):
        o = list(range(1, N+1)); rng.shuffle(o); base_orders.append(o)
    for o in base_orders:
        if time.time() - start > DEADLINE - 8:
            break
        S = _grow([], o)
        consider(S)

    # Phase 2: shake-and-regrow local search from best
    hard = DEADLINE - 3
    while time.time() - start < hard:
        k = rng.randint(1, 5)
        shake = rng.sample(best, min(k, len(best)))
        o = list(range(1, N+1)); rng.shuffle(o)
        try_start(o, shake)
        if time.time() - start < hard:
            o2 = list(range(N, 0, -1))
            try_start(o2, None)

    return best
