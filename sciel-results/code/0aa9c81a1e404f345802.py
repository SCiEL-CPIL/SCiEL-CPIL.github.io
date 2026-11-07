<<<<<<< SEARCH
    # Precompute forbidden primes for each gap size
    # forbidden_primes[g] is a tuple of primes p such that p < g
    forbidden_primes = {}
    for g in range(2, max_gap + 1):
        fp = tuple(p for p in primes_up_to_max_gap if p < g)
        forbidden_primes[g] = fp

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break

        gap = hi - lo
        fp = forbidden_primes[gap]

        # If no forbidden primes, any number is valid
        if not fp:
            result[i] = lo + 1
            continue

        n = hi - lo - 1  # number of candidates
        if n <= 0:
            continue

        # Use NumPy to mark invalid candidates in bulk
        valid = np.ones(n, dtype=bool)

        for p in fp:
            # Find first multiple of p >= lo+1
            first = ((lo + 1 + p - 1) // p) * p
            # Calculate indices of multiples in [lo+1, hi-1]
            # Indices are m - (lo+1)
            start_idx = first - (lo + 1)
            if start_idx < 0:
                start_idx = 0
            
            # Mark as invalid
            valid[start_idx::p] = False

        # Find first valid candidate
        if np.any(valid):
            idx = np.argmax(valid)
            result[i] = int(lo + 1 + idx)
        else:
            # Fallback: trial division (should not happen for these gaps)
            for m in range(lo + 1, hi):
                is_valid = True
                for p in fp:
                    if m % p == 0:
                        is_valid = False
                        break
                if is_valid:
                    result[i] = m
                    break
=======
    # Precompute forbidden primes for each gap size
    forbidden_primes = {}
    for g in range(2, max_gap + 1):
        fp = tuple(p for p in primes_up_to_max_gap if p < g)
        forbidden_primes[g] = fp

    # Precompute a bitmask per distinct gap size.
    # For gap size g with n = g-1 candidate offsets (0..n-1),
    # bitmask has bit k set iff offset k is valid (not divisible by any forbidden prime).
    gap_bitmasks = {}
    for g in set(hi - lo for lo, hi in P['gaps']):
        n = g - 1
        if n <= 0:
            gap_bitmasks[g] = 0
            continue
        # Start with all bits set
        bm = (1 << n) - 1
        fp = forbidden_primes[g]
        for p in fp:
            # Create a mask with bits set at positions that are multiples of p
            # within [0, n-1]
            invalid_mask = 0
            for m in range(0, n, p):
                invalid_mask |= (1 << m)
            bm &= ~invalid_mask
        gap_bitmasks[g] = bm

    result = [None] * len(P['gaps'])

    for i, (lo, hi) in enumerate(P['gaps']):
        if time.time() > deadline:
            break

        gap = hi - lo
        n = gap - 1
        if n <= 0:
            continue

        bm = gap_bitmasks[gap]
        if bm == 0:
            # No valid candidate; fallback to trial division
            fp = forbidden_primes[gap]
            for m in range(lo + 1, hi):
                is_valid = True
                for p in fp:
                    if m % p == 0:
                        is_valid = False
                        break
                if is_valid:
                    result[i] = m
                    break
            continue

        # Find the least significant set bit in bm
        # This gives the first valid offset
        offset = (bm & (-bm)).bit_length() - 1
        result[i] = lo + 1 + offset
>>>>>>> REPLACE
