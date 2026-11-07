"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_perfect_square(n):
    if n < 0:
        return False
    r = math.isqrt(n)
    return r * r == n

def _squarefree_kernel(n):
    """Return the squarefree kernel of n as a frozenset of primes with odd exponent."""
    factors = {}
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            exp = 0
            while temp % d == 0:
                temp //= d
                exp += 1
            if exp % 2 == 1:
                factors[d] = 1
        d += 1
    if temp > 1:
        factors[temp] = 1
    return frozenset(factors.keys())

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 210
    MAX_VAL = 10210
    MAX_SIZE = 100

    # Precompute kernels for all numbers in (210, MAX_VAL]
    kernels = {}
    for x in range(N + 1, MAX_VAL + 1):
        kernels[x] = _squarefree_kernel(x)

    # Collect all primes that appear in any kernel or in target
    all_primes_set = set()
    for k in kernels.values():
        all_primes_set.update(k)
    all_primes_set.update([2, 3, 5, 7])
    all_primes = sorted(all_primes_set)
    num_primes = len(all_primes)
    prime_to_idx = {p: i for i, p in enumerate(all_primes)}

    # Build target mask: XOR of selected kernels must equal kernel of 210 = {2,3,5,7}
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_to_idx[p])

    # Build bitmask for each number
    vector_masks = {}
    for x in range(N + 1, MAX_VAL + 1):
        mask = 0
        for p in kernels[x]:
            mask |= (1 << prime_to_idx[p])
        vector_masks[x] = mask

    # Incremental Gaussian elimination over GF(2) using bitmasks
    # basis: dict from pivot_bit -> (mask, frozenset_of_numbers)
    basis = {}

    best_result = None

    for M in range(N + 1, MAX_VAL + 1):
        if time.time() > deadline:
            break

        # Add vector for number M to the basis
        mask = vector_masks[M]
        m = mask
        combo = {M}
        while m != 0:
            pc = m.bit_length() - 1
            if pc in basis:
                bm, bc = basis[pc]
                m ^= bm
                combo = combo.symmetric_difference(bc)
            else:
                basis[pc] = (m, frozenset(combo))
                break
        # If m == 0, the vector was already in the span; no new basis vector added

        # Check if target is in the span of {211, ..., M}
        tm = target_mask
        tcombo = set()
        in_span = True
        while tm != 0:
            pc = tm.bit_length() - 1
            if pc not in basis:
                in_span = False
                break
            bm, bc = basis[pc]
            tm ^= bm
            tcombo = tcombo.symmetric_difference(bc)

        if in_span and tm == 0:
            # Found a valid combination
            if 1 <= len(tcombo) <= MAX_SIZE:
                # Verify the product
                prod = N
                for x in tcombo:
                    prod *= x
                if _is_perfect_square(prod):
                    best_result = sorted(tcombo)
                    break

    if best_result is None:
        # Fallback: 210 * 840 = 210 * 210 * 4 = (210*2)^2 = 420^2
        best_result = [840]

    # Ensure return type is list of Python ints
    return [int(x) for x in best_result]
# EVOLVE-BLOCK-END
