import time
from math import gcd
from collections import defaultdict

N = 512
DEADLINE = 75.0  # safety margin under 90s kill

def _factorize(n):
    d = {}
    x = n
    p = 2
    while p * p <= x:
        while x % p == 0:
            d[p] = d.get(p, 0) + 1
            x //= p
        p += 1
    if x > 1:
        d[x] = d.get(x, 0) + 1
    return d

def _lcm(a, b):
    return a // gcd(a, b) * b

def construct():
    t0 = time.time()

    # factorizations and pairwise conflict bitsets
    facs = [None] + [_factorize(i) for i in range(1, N + 1)]
    conf = [0] * (N + 1)
    for a in range(1, N + 1):
        fa = facs[a]
        bits = 0
        for b in range(a + 1, N + 1):
            if _lcm(a, b) > N:
                bits |= (1 << b)
        conf[a] = bits
        conf[b] = bits  # will be overwritten correctly below

    # rebuild symmetrically and correctly
    conf = [0] * (N + 1)
    for a in range(1, N + 1):
        fa = facs[a]
        bits = 0
        for b in range(a + 1, N + 1):
            fb = facs[b]
            ok = True
            for p in fa:
                e = fa[p]
                if fb.get(p, 0) + e > 9 if p == 2 else (fb.get(p, 0) + e > 1 if p == 3 else (fb.get(p, 0) + e > 0)):
                    # need product p^max <= 512; check via direct bound
                    pass
            # direct check is safest
            if _lcm(a, b) > N:
                bits |= (1 << b)
                conf[b] |= (1 << a)
        conf[a] |= bits

    def size(S):
        return S.bit_count()

    def valid_add(S, x):
        return (conf[x] & S) == 0

    def valid_remove(S, x):
        return S & (1 << x)

    def can_add(S, x):
        return (conf[x] & S) == 0

    # ---- seed: divisors of 512 (powers of 2) ----
    S = 0
    for i in range(1, N + 1):
        if N % i == 0:
            S |= (1 << i)
    best = S

    # ---- greedy fill by increasing conflict degree ----
    def degree(x):
        return conf[x].bit_count()

    order = sorted(range(1, N + 1), key=lambda x: degree(x))
    changed = True
    while changed and time.time() - t0 < DEADLINE:
        changed = False
        for x in order:
            if not (S & (1 << x)) and can_add(S, x):
                S |= (1 << x)
                changed = True
    if size(S) > size(best):
        best = S

    # ---- local search: 2-swaps, k-insertions, k-deletions (bitset) ----
    def try_insert(S):
        # add any single element that fits
        for x in range(1, N + 1):
            if not (S & (1 << x)) and can_add(S, x):
                return S | (1 << x)
        return S

    def try_delete(S):
        for x in range(1, N + 1):
            if S & (1 << x):
                return S & ~(1 << x)
        return S

    def two_swap(S):
        # remove one, add two
        for a in range(1, N + 1):
            if not (S & (1 << a)):
                continue
            S1 = S & ~(1 << a)
            # find two to add
            cand = []
            for x in range(1, N + 1):
                if not (S1 & (1 << x)) and can_add(S1, x):
                    cand.append(x)
                    if len(cand) >= 2:
                        break
            if len(cand) >= 2:
                b, c = cand[0], cand[1]
                S2 = S1 | (1 << b) | (1 << c)
                if can_add(S1 | (1 << b), c):
                    return S2
        return S

    def three_swap(S):
        # remove two, add three
        for a in range(1, N + 1):
            if not (S & (1 << a)):
                continue
            for b in range(1, N + 1):
                if not (S & (1 << b)) or a == b:
                    continue
                S1 = S & ~(1 << a) & ~(1 << b)
                cand = []
                for x in range(1, N + 1):
                    if not (S1 & (1 << x)) and can_add(S1, x):
                        cand.append(x)
                        if len(cand) >= 3:
                            break
                if len(cand) >= 3:
                    c, d, e = cand[0], cand[1], cand[2]
                    if can_add(S1 | (1 << c), d) and can_add(S1 | (1 << c) | (1 << d), e):
                        return S1 | (1 << c) | (1 << d) | (1 << e)
        return S

    # iterate improvements
    improved = True
    while improved and time.time() - t0 < DEADLINE:
        improved = False
        cur = best
        # single insert
        ns = try_insert(cur)
        if size(ns) > size(cur):
            cur = ns
            improved = True
        # 2-swap
        ns = two_swap(cur)
        if size(ns) > size(cur):
            cur = ns
            improved = True
        # 3-swap
        if time.time() - t0 < DEADLINE:
            ns = three_swap(cur)
            if size(ns) > size(cur):
                cur = ns
                improved = True
        if size(cur) > size(best):
            best = cur

    # ---- final polish: greedy fill once more ----
    order = sorted(range(1, N + 1), key=lambda x: degree(x))
    for x in order:
        if not (best & (1 << x)) and can_add(best, x):
            best |= (1 << x)

    result = [i for i in range(1, N + 1) if best & (1 << i)]
    return result
