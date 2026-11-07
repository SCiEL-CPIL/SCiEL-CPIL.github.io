"""Square-product extension for 210: find distinct ints in (210,10210]
whose product * 210 is a perfect square, minimizing max - 210."""

import math
import time
from collections import defaultdict

N = 210
MAX_N = 10210
MAX_K = 100
SMALL_PRIMES = [2, 3, 5, 7]


def _is_square(n):
    r = math.isqrt(n)
    return r * r == n


def _sieve_primes(limit):
    sieve = [True] * (limit + 1)
    if limit >= 0:
        sieve[0] = False
    if limit >= 1:
        sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]


def _factorize(n, primes):
    d = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            e = 0
            while n % p == 0:
                n //= p
                e += 1
            d[p] = e
    if n > 1:
        d[n] = 1
    return d


def construct():
    t_start = time.time()
    DEADLINE = t_start + 75.0  # safety margin before 90s kill

    primes = _sieve_primes(MAX_N)

    # Precompute parity for all candidates
    # Parity vector: (small_mask, frozenset_of_large_odd_primes)
    parity = {}
    for n in range(211, MAX_N + 1):
        d = _factorize(n, primes)
        mask = 0
        for i, p in enumerate(SMALL_PRIMES):
            if d.get(p, 0) & 1:
                mask |= (1 << i)
        large = frozenset(p for p, e in d.items() if p not in SMALL_PRIMES and e & 1)
        parity[n] = (mask, large)

    TARGET_MASK = 0b1111  # 210 = 2*3*5*7, all odd
    TARGET_LARGE = frozenset()
    TARGET_VECTOR = (TARGET_MASK, TARGET_LARGE)

    # Incremental search: add elements one by one from 211 upward
    # Maintain:
    #   candidates: list of (n, vec) for all added elements
    #   pair_dict: maps pair_xor_vec -> (i, j) indices into candidates
    #   For 3-element: when adding new element k, check if (TARGET ^ vec_k) is in pair_dict
    #   For 2-element: check if (TARGET ^ vec_k) equals any previous vec
    #   For 1-element: check if vec_k == TARGET

    candidates = []  # list of (n, vec)
    # For 2-element check: map vec -> index in candidates
    single_dict = {}  # vec -> index
    # For 3-element check: map pair_xor_vec -> (i, j)
    pair_dict = {}  # (mask, frozenset) -> (i, j)

    for limit in range(211, MAX_N + 1):
        if time.time() > DEADLINE:
            break

        v = parity[limit]
        idx = len(candidates)
        candidates.append((limit, v))

        # Check 1-element
        if v == TARGET_VECTOR:
            return [limit]

        # Check 2-element: need prev_vec such that prev_vec ^ v == TARGET
        # i.e., prev_vec == TARGET ^ v
        req_single = (TARGET_MASK ^ v[0], TARGET_LARGE.symmetric_difference(v[1]))
        if req_single in single_dict:
            prev_idx = single_dict[req_single]
            n_prev = candidates[prev_idx][0]
            return [n_prev, limit]

        # Check 3-element: need pair (i,j) such that vec_i ^ vec_j == TARGET ^ v
        # i.e., pair_xor == TARGET ^ v
        req_pair = (TARGET_MASK ^ v[0], TARGET_LARGE.symmetric_difference(v[1]))
        if req_pair in pair_dict:
            i, j = pair_dict[req_pair]
            n1 = candidates[i][0]
            n2 = candidates[j][0]
            return [n1, n2, limit]

        # Now add this element to the dictionaries
        # Add to single_dict (first occurrence is fine for finding any valid pair)
        if v not in single_dict:
            single_dict[v] = idx

        # Add all pairs (idx, k) for k < idx to pair_dict
        for k in range(idx):
            n_k, v_k = candidates[k]
            # Compute v_k ^ v
            m_xor = v_k[0] ^ v[0]
            if v_k[1] == v[1]:
                lg_xor = frozenset()
            else:
                lg_xor = v_k[1].symmetric_difference(v[1])
            pair_key = (m_xor, lg_xor)
            # Store first occurrence only (any valid pair works)
            if pair_key not in pair_dict:
                pair_dict[pair_key] = (k, idx)

    # Fallback: shouldn't happen, but return something valid
    # 210 * 840 = 210 * 210 * 4 = (210*2)^2 = 420^2
    return [840]


if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Result: {result}")
    print(f"Max - 210: {max(result) - 210}")
    prod = 210
    for x in result:
        prod *= x
    print(f"Product is square: {_is_square(prod)}")
    print(f"Time: {t1 - t0:.2f}s")
