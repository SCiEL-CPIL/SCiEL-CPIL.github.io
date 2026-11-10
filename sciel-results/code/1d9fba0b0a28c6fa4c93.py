"""Combined program: seed with contiguous prefix, then greedy with multiple orderings, keep best."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = start + 85.0  # safety margin under 90s kill

    LIMIT = 512

    def can_add(x, S):
        for a in S:
            if x * a // gcd(x, a) > LIMIT:
                return False
        return True

    def max_contiguous_k(L):
        k = 1
        while True:
            nk = k + 1
            ok = True
            for a in range(1, k + 1):
                if nk * a // gcd(nk, a) > L:
                    ok = False
                    break
            if not ok:
                break
            k = nk
            if time.time() > DEADLINE:
                return k
        return k

    k = max_contiguous_k(LIMIT)
    base_S = set(range(1, k + 1))
    candidates = list(range(k + 1, LIMIT + 1))

    def greedy(order):
        S = set(base_S)
        for x in order:
            if time.time() > DEADLINE:
                break
            if can_add(x, S):
                S.add(x)
        return S

    # Multiple orderings; keep the largest resulting set.
    orderings = []

    # 1. numerical ascending
    orderings.append(sorted(candidates))

    # 2. largest prime factor ascending, then value
    def lpf(n):
        m = n
        f = 2
        while f * f <= m:
            while m % f == 0:
                m //= f
            f += 1
        return m if m > 1 else f
    lpf_arr = [lpf(x) for x in candidates]
    orderings.append(sorted(candidates, key=lambda x: (lpf[x - (k + 1)], x)))

    # 3. smallest value of max_{a in base} lcm(x,a) (most compatible with base first)
    def base_max_lcm(x):
        mx = 1
        for a in base_S:
            v = x * a // gcd(x, a)
            if v > mx:
                mx = v
        return mx
    orderings.append(sorted(candidates, key=lambda x: (base_max_lcm(x), x)))

    # 4. largest prime factor descending (add "hard" ones early while set is small)
    orderings.append(sorted(candidates, key=lambda x: (-lpf[x - (k + 1)], x)))

    # 5. numerical descending
    orderings.append(sorted(candidates, reverse=True))

    best = set(base_S)
    for order in orderings:
        if time.time() > DEADLINE:
            break
        S = greedy(order)
        if len(S) > len(best):
            best = S

    return sorted(best)
# EVOLVE-BLOCK-END
