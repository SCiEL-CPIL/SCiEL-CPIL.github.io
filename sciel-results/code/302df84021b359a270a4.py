import math
import time

def construct():
    limit = 10210
    start_time = time.time()
    deadline = start_time + 52.0

    # Precompute smallest prime factor
    spf = list(range(limit + 1))
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    nprimes = len(primes)

    # Precompute parity bitmasks for all n in [211, limit]
    bitmasks = {}
    for n in range(211, limit + 1):
        mask = 0
        x = n
        while x > 1:
            p = spf[x]
            cnt = 0
            while x % p == 0:
                x //= p
                cnt += 1
            if cnt % 2 == 1:
                mask |= (1 << prime_to_idx[p])
        bitmasks[n] = mask

    # Target mask for 210
    target_mask = 0
    x = 210
    while x > 1:
        p = spf[x]
        cnt = 0
        while x % p == 0:
            x //= p
            cnt += 1
        if cnt % 2 == 1:
            target_mask |= (1 << prime_to_idx[p])

    # Binary search on answer: find minimum M such that a valid subset of [211, M] exists
    # with product*210 being a perfect square.
    # We binary search on M. For each candidate M, we check if target_mask is in the
    # span of {bitmasks[211], ..., bitmasks[M]} over GF(2).
    # If yes, we extract a solution.

    # To speed up, we precompute the basis incrementally and store snapshots.
    # But for simplicity and correctness, we do a binary search and for each M,
    # we run Gaussian elimination on [211, M].
    # However, that's O(log(limit)) * O(limit * nprimes) which might be too slow.
    # Instead, we do a linear scan from 211 upward, maintaining the basis, and
    # check at each step if target is in the span. The first M where this happens
    # is the minimum.

    # Actually, the linear scan IS the minimal M. Let's just do the linear scan
    # but with a more efficient basis representation.

    # Use a list-based basis for speed
    # basis: list of (pivot, vec, comb) where vec is the bitmask and comb is the combination bitmask
    # We maintain the basis in reduced row echelon form with pivots sorted descending.

    basis_vecs = {}  # pivot -> (vec, comb)
    pivots = []  # sorted descending
    t_u = target_mask
    t_c = 0

    for M in range(211, limit + 1):
        if time.time() > deadline:
            break

        v = bitmasks[M]
        c = 1 << (M - 211)

        # Reduce v
        for p in pivots:
            if (v >> p) & 1:
                bv, bc = basis_vecs[p]
                v ^= bv
                c ^= bc

        if v != 0:
            pivot = v.bit_length() - 1
            basis_vecs[pivot] = (v, c)
            # Insert pivot into sorted list (descending)
            # Find position
            pos = len(pivots)
            for i in range(len(pivots)):
                if pivots[i] < pivot:
                    pos = i
                    break
            pivots.insert(pos, pivot)

            # Update target
            if (t_u >> pivot) & 1:
                t_u ^= v
                t_c ^= c

        # Check if target is satisfied
        if t_u == 0:
            # Extract solution
            selected = []
            for i in range(M - 211 + 1):
                if (t_c >> i) & 1:
                    selected.append(211 + i)

            if 1 <= len(selected) <= 100:
                # Verify
                prod = 210
                for x in selected:
                    prod *= x
                s = math.isqrt(prod)
                if s * s == prod:
                    return [int(x) for x in selected]

    # Fallback
    return [840]
