"""Search for A,B (3 distinct ints in [1,128]) maximizing #distinct sums in A+B
that are pairwise coprime."""
import time
import random
from math import gcd

_START = time.time()
_LIMIT = 82.0  # safety margin under the 90s wall-clock kill

# All primes up to 128 (a prime p divides a sum value s<=256 iff p|s; for s>128
# the only possible prime divisors relevant to coprimality of two sums are <=128
# because any common prime factor of two sums <=256 is <=256, but if it is >128
# it can divide at most one sum in [1,256] unless the sum is 2*p, so we include
# primes up to 128 which covers all shared factors for values in [1,256] except
# the special 2*p cases; gcd check below is authoritative anyway.)
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59,
           61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127]
_SMALL = _PRIMES[:10]  # for cheap early rejection

def _distinct_sums(A, B):
    return sorted(set(a + b for a in A for b in B))

def _pairwise_coprime(vals):
    n = len(vals)
    for i in range(n):
        vi = vals[i]
        for j in range(i + 1, n):
            if gcd(vi, vals[j]) != 1:
                return False
    return True

def _score(A, B):
    vals = _distinct_sums(A, B)
    if _pairwise_coprime(vals):
        return len(vals)
    return 0

def _valid(A, B):
    return (len(set(A)) == 3 and len(set(B)) == 3
            and all(1 <= x <= 128 for x in A) and all(1 <= x <= 128 for x in B))

def _improve_B(A, best_B, best_val):
    """Local search / hill climbing over B to maximize coprime distinct sums."""
    cur_B = list(best_B)
    cur_val = best_val
    for _ in range(6000):
        if time.time() > _LIMIT:
            break
        i = random.randrange(3)
        new = random.randint(1, 128)
        if new == cur_B[i]:
            continue
        old = cur_B[i]
        cur_B[i] = new
        # distinctness of B required
        if len(set(cur_B)) != 3:
            cur_B[i] = old
            continue
        vals = _distinct_sums(A, cur_B)
        if _pairwise_coprime(vals):
            v = len(vals)
        else:
            v = 0
        if v >= cur_val:
            cur_val = v
        else:
            cur_B[i] = old
    return cur_B, cur_val

def _full_search():
    best = [[1, 2, 3], [1, 2, 3]]
    best_val = 0
    n = 128
    # iterate A as a 3-combination (order irrelevant for sumset)
    for a1 in range(1, n + 1):
        if time.time() > _LIMIT:
            break
        for a2 in range(a1 + 1, n + 1):
            for a3 in range(a2 + 1, n + 1):
                if time.time() > _LIMIT:
                    break
                A = (a1, a2, a3)
                # seed B with a few diverse starts, then hill climb
                seeds = [
                    (1, 2, 3),
                    (1, 2, 4),
                    (a1, a2, a3),
                    (n - a1, n - a2, n - a3),
                    (1, 64, 128),
                    (random.randint(1, n), random.randint(1, n), random.randint(1, n)),
                ]
                for sb in seeds:
                    if time.time() > _LIMIT:
                        break
                    B0 = [x for x in sb]
                    if len(set(B0)) != 3:
                        continue
                    if not all(1 <= x <= n for x in B0):
                        continue
                    B, v = _improve_B(A, B0, 0)
                    if v > best_val:
                        best_val = v
                        best = [list(A), list(B)]
                        if best_val == 9:
                            return best, best_val
    return best, best_val

def _verify(A, B):
    if not _valid(A, B):
        return 0
    return _score(A, B)

def construct():
    # deterministic baseline
    baseline_A = [1, 2, 3]
    baseline_B = [1, 2, 3]
    best_val = _verify(baseline_A, baseline_B)
    best = [list(baseline_A), list(baseline_B)]

    # A few quick targeted seeds to seed the search
    random.seed(12345)
    quick_seeds = [
        ([1, 2, 3], [1, 2, 3]),
        ([1, 2, 3], [1, 2, 4]),
        ([1, 2, 4], [1, 2, 3]),
        ([2, 3, 5], [1, 2, 3]),
        ([1, 3, 5], [1, 2, 3]),
        ([1, 2, 3], [2, 3, 5]),
    ]
    for A, B in quick_seeds:
        if time.time() > _LIMIT:
            break
        B, v = _improve_B(A, list(B), 0)
        if v > best_val:
            best_val = v
            best = [list(A), list(B)]
            if best_val == 9:
                return best

    # Exhaustive-ish over A with B hill-climbing
    best, best_val = _full_search()

    # Final polish: try to push toward 9 with extra random restarts if time remains
    if best_val < 9 and time.time() < _LIMIT - 2:
        for _ in range(200):
            if time.time() > _LIMIT:
                break
            A = random.sample(range(1, 129), 3)
            B = random.sample(range(1, 129), 3)
            B, v = _improve_B(A, B, 0)
            if v > best_val:
                best_val = v
                best = [list(A), list(B)]
                if best_val == 9:
                    break

    # ensure validity; fall back to a known-valid pair
    if not _valid(best[0], best[1]) or _verify(best[0], best[1]) == 0:
        # fallback: find any valid pair with >=1 distinct coprime sum
        for A, B in quick_seeds:
            if _verify(A, B) > 0:
                return [list(A), list(B)]
        return [[1, 2, 3], [1, 2, 3]]

    return [list(best[0]), list(best[1])]
