import time
from math import isqrt

def _is_perfect_square(n):
    if n < 0:
        return False
    r = isqrt(n)
    return r * r == n

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    max_val = 10210
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_to_idx = {p: i for i, p in enumerate(primes)}

    def get_mask(n):
        mask = 0
        x = n
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                cnt = 0
                while x % p == 0:
                    x //= p
                    cnt ^= 1
                if cnt:
                    mask |= (1 << prime_to_idx[p])
        if x > 1:
            if x in prime_to_idx:
                mask |= (1 << prime_to_idx[x])
        return mask

    target_mask = get_mask(210)

    masks = [0] * (max_val + 1)
    for i in range(211, max_val + 1):
        masks[i] = get_mask(i)

    def check(M_max):
        b = {}
        for M in range(211, M_max + 1):
            m = masks[M]
            cur = m
            used = frozenset([M])
            while cur != 0:
                hb = cur.bit_length() - 1
                if hb in b:
                    bm, bu = b[hb]
                    cur ^= bm
                    used = used.symmetric_difference(bu)
                else:
                    break
            if cur != 0:
                hb = cur.bit_length() - 1
                b[hb] = (cur, used)

        cur = target_mask
        used = frozenset()
        while cur != 0:
            hb = cur.bit_length() - 1
            if hb in b:
                bm, bu = b[hb]
                cur ^= bm
                used = used.symmetric_difference(bu)
            else:
                return None

        if len(used) > 100:
            return None

        prod = 210
        for x in used:
            prod *= x
        if _is_perfect_square(prod):
            return sorted(used)
        return None

    lo, hi = 211, max_val
    best_result = None

    r = check(max_val)
    if r is not None:
        best_result = r

    while lo < hi and time.time() < deadline - 3.0:
        mid = (lo + hi) // 2
        r = check(mid)
        if r is not None:
            best_result = r
            hi = mid
        else:
            lo = mid + 1

    if best_result is not None:
        return [int(x) for x in best_result]

    return [224, 240]
