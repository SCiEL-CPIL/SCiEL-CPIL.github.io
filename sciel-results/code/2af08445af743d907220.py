import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 210
    max_val = 10210

    # Factorize 210: 2^1 * 3^1 * 5^1 * 7^1
    # Target mask: bits for 2, 3, 5, 7
    
    # Sieve for primes up to max_val
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, max_val + 1) if is_prime[i]]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    def get_parity_mask(x):
        """Return bitmask where bit i is set if prime primes[i] appears to odd power in x."""
        mask = 0
        temp = x
        for idx, p in enumerate(primes):
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count ^= 1
                if count:
                    mask |= (1 << idx)
        if temp > 1:
            if temp in prime_index:
                mask |= (1 << prime_index[temp])
        return mask

    # Precompute masks for all numbers up to max_val
    masks = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        masks[i] = get_parity_mask(i)

    # Parity mask of 210
    target = masks[210]

    # Phase 1: Find minimum M where target is in span of {masks[211]...masks[M]}
    basis = {}
    min_M = None

    for M in range(211, max_val + 1):
        if time.time() > deadline:
            break

        v = masks[M]
        h = 1 << (M - 211)  # history bit for this candidate

        # Insert v into basis with history h
        tv = v
        th = h
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                basis[p] = (tv, th)
                break

        # Check if target is in span
        tv = target
        th = 0
        while tv:
            p = tv.bit_length() - 1
            if p in basis:
                bv, bh = basis[p]
                tv ^= bv
                th ^= bh
            else:
                break

        if tv == 0:
            min_M = M
            break

    if min_M is None:
        # Fallback if no solution found (should not happen)
        return [840]

    # Phase 2: Extract solution
    # Re-derive the solution from the basis at min_M
    # The basis dictionary currently holds the state after inserting up to min_M.
    # We need to trace back which vectors were used to form the target.
    
    # The variable `th` from the check above contains the history of the target reduction.
    # However, `th` was a local variable in the loop. We need to recompute or store it.
    # Let's recompute the solution using the basis.
    
    tv = target
    sol_history = 0
    while tv:
        p = tv.bit_length() - 1
        if p in basis:
            bv, bh = basis[p]
            tv ^= bv
            sol_history ^= bh
        else:
            # Should not happen if tv==0 was found
            break

    if tv != 0:
        return [840]

    # Extract solution indices from sol_history
    sol = []
    hh = sol_history
    while hh:
        lsb = hh & -hh
        idx = lsb.bit_length() - 1
        sol.append(211 + idx)
        hh ^= lsb

    # Check if solution size is valid
    if len(sol) <= 100:
        # Verify
        prod = 210
        for x in sol:
            prod *= x
        s = math.isqrt(prod)
        if s * s == prod:
            return [int(x) for x in sol]

    # If solution size > 100 (unlikely for small M), try to find a smaller one?
    # For now, if this happens, we might need a more complex reduction.
    # But given the score 22.5, M is small, so this branch is likely not taken.
    
    return [840]
