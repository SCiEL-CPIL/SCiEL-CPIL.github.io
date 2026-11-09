"""Segmented sieve for sigma(n), group by sigma, two-pointer exact search."""
import time
from collections import defaultdict

MAXN = 10_000_000
BLOCK = 1_000_000
DEADLINE_MARGIN = 5.0  # seconds to leave before the 60s kill

# ---------- segmented sigma ----------
def _seg_sigma(n, block_size, deadline):
    """Return list sigma[1..n]."""
    n = int(n)
    sigma = [0] * (n + 1)
    sigma[0] = 0
    sigma[1] = 1

    # smallest prime factor up to sqrt(n)
    lim = int(n ** 0.5) + 1
    spf = [0] * (lim + 1)
    for i in range(2, lim + 1):
        if spf[i] == 0:
            spf[i] = i
            for j in range(i * i, lim + 1, i):
                if spf[j] == 0:
                    spf[j] = i
    primes = [p for p in range(2, lim + 1) if spf[p] == p]

    for lo in range(1, n + 1, block_size):
        hi = min(lo + block_size - 1, n)
        seg_sigma = [1] * (hi - lo + 1)
        for p in primes:
            if p * p > hi:
                break
            if time.time() > deadline:
                break
            # smallest multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            # accumulate p^k terms
            pk = p
            while pk <= hi:
                for m in range(start, hi + 1, pk):
                    seg_sigma[m - lo] *= (1 + p)
                pk *= p
                if pk > hi:
                    break
        for i in range(hi - lo + 1):
            if seg_sigma[i] == 1:
                # no prime factor <= sqrt => the number itself is prime (or 1)
                v = lo + i
                seg_sigma[i] = v + 1
        sigma[lo:hi + 1] = seg_sigma
    return sigma

# ---------- exact two-pointer per sigma group ----------
def _search_groups(groups, deadline):
    """groups: dict sigma -> list of n. Return (best_err_num, best_err_den, best_a, best_b)."""
    best_num = None
    best_den = 1
    best_a = best_b = 0
    for s, arr in groups.items():
        if len(arr) < 2:
            continue
        if time.time() > deadline:
            break
        arr.sort()
        m = len(arr)
        b = 0
        for a in arr:
            # want b ≈ (2/3) a
            target = (2 * a) // 3
            # advance b until b >= target
            while b + 1 < m and arr[b + 1] < target:
                b += 1
            # evaluate b and b+1
            for idx in (b, b + 1):
                if idx >= m:
                    continue
                bv = arr[idx]
                num = abs(2 * a - 3 * bv)
                den = 2 * bv
                if best_num is None:
                    best_num, best_den, best_a, best_b = num, den, a, bv
                else:
                    # num/den < best_num/best_den  <=>  num*best_den < best_num*den
                    if num * best_den < best_num * den:
                        best_num, best_den, best_a, best_b = num, den, a, bv
    return best_num, best_den, best_a, best_b

def construct():
    t0 = time.time()
    deadline = t0 + 60.0 - DEADLINE_MARGIN

    # initial safe fallback (sigma(11)=12, sigma(6)=12)
    best_num, best_den, best_a, best_b = None, 1, 11, 6

    try:
        sigma = _seg_sigma(MAXN, BLOCK, deadline)
        if time.time() > deadline:
            return [best_a, best_b]

        groups = defaultdict(list)
        for i in range(1, MAXN + 1):
            groups[sigma[i]].append(i)

        if time.time() > deadline:
            return [best_a, best_b]

        best_num, best_den, best_a, best_b = _search_groups(groups, deadline)
    except Exception:
        pass

    if best_num is None:
        return [11, 6]

    # ensure types are exactly Python int
    best_a = int(best_a)
    best_b = int(best_b)
    if best_a <= 0 or best_b <= 0 or best_a > MAXN or best_b > MAXN:
        return [11, 6]
    # raw value must be nonzero
    if best_num == 0:
        # find a distinct pair with same sigma if possible, else fallback
        return [11, 6]
    return [best_a, best_b]
