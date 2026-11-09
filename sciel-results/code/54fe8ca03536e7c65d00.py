    def compute_phi_segment(lo, hi, primes):
        n = hi - lo + 1
        phi = np.arange(lo, hi + 1, dtype=np.int64)
        rem = np.arange(lo, hi + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > hi:
                break
            start = ((lo + p - 1) // p) * p
            # Indices of multiples of p in [lo, hi]
            indices = np.arange(start, hi + 1, p)
            # Update phi
            phi[indices] -= phi[indices] // p
            # Update rem
            # Divide by p as long as divisible
            # Since p is small, we can do a few iterations
            # Or just divide once and check?
            # If p^2 <= hi, then p^2 might divide m.
            # We need to remove all factors of p from rem.
            # rem[indices] //= p
            # But some might still be divisible.
            # We can loop while any(rem[indices] % p == 0)
            # But this is expensive.
            # Note: p <= sqrt(hi). So p^2 <= hi.
            # Max exponent of p in m is log_p(hi).
            # For p=2, hi=10^12, exponent is 40.
            # So we need up to 40 divisions.
            
            # Efficient way:
            mask = (indices >= lo) & (indices <= hi) # already true
            # We can just do:
            # while True:
            #    div_mask = (rem[indices] % p == 0)
            #    if not np.any(div_mask): break
            #    rem[indices[div_mask]] //= p
            
            # This is still a python loop inside the prime loop.
            # But the number of iterations is small (log_p(hi)).
            
            # Let's try to do it in one go if possible?
            # No, variable number of divisions.
            
            # Optimization:
            # For most numbers, p divides only once.
            # rem[indices] //= p
            # Then check if any are still divisible.
            # If so, divide again.
            
            rem_vals = rem[indices]
            rem_vals //= p
            # Check if any still divisible
            while np.any(rem_vals % p == 0):
                rem_vals[rem_vals % p == 0] //= p
            rem[indices] = rem_vals
            
        # Correction for large prime factors
        # For each m, if rem[m] > 1, it is a prime factor q.
        # phi[m] -= phi[m] // rem[m]
        large_primes_mask = rem > 1
        if np.any(large_primes_mask):
            phi[large_primes_mask] -= phi[large_primes_mask] // rem[large_primes_mask]
            
        return phi
    