"""Constructs integer n <= 10^18 maximizing tau_perp(n):
the number of adjacent coprime pairs in the sorted divisor list.

Strategy: square-free numbers (product of distinct primes) are optimal because
higher exponents cluster divisors and reduce the coprime-adjacent fraction.
We enumerate subsets of the first several primes (size >= 11, product <= 10^18),
compute tau_perp exactly, and return the best.
"""
import time
import itertools

_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
_LIMIT = 10**18


def _tau_perp_square_free(primes):
    k = len(primes)
    pairs = [(1, 0)]
    for i, p in enumerate(primes):
        pairs += [(v * p, m | (1 << i)) for (v, m) in pairs]
    pairs.sort()
    cnt = 0
    for i in range(len(pairs) - 1):
        if pairs[i][1] & pairs[i + 1][1] == 0:
            cnt += 1
    return cnt


def construct():
    deadline = time.time() + 140

    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _tau_perp_square_free([2, 3, 5, 7, 11, 13, 17, 19])

    npr = len(_PRIMES)
    # enumerate subsets of size >= 11
    for r in range(11, npr + 1):
        for combo in itertools.combinations(range(npr), r):
            if time.time() > deadline:
                return best_n
            prod = 1
            for idx in combo:
                prod *= _PRIMES[idx]
                if prod > _LIMIT:
                    break
            if prod > _LIMIT:
                continue
            primes = [_PRIMES[idx] for idx in combo]
            sc = _tau_perp_square_free(primes)
            if sc > best_score:
                best_score = sc
                best_n = prod

    return best_n
