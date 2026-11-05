    def compute_phi_segment(L, R):
        size = R - L + 1
        phi = np.arange(L, R + 1, dtype=np.int64)
        remaining = np.arange(L, R + 1, dtype=np.int64)
        
        for p in primes:
            if p * p > R:
                break
            # Find indices of multiples of p in [L, R]
            # start = ceil(L/p) * p
            start = ((L + p - 1) // p) * p
            if start > R:
                continue
            
            # Indices in the array [0, size-1] corresponding to multiples
            # j = start, start+p, ..., <= R
            # idx = j - L
            # idx starts at start - L, steps by p
            idxs = np.arange(start - L, size, p)
            
            # Apply phi factor
            phi[idxs] -= phi[idxs] // p
            
            # Remove p from remaining
            # This is tricky to vectorize with "while" logic.
            # But we can just do:
            # remaining[idxs] //= p  (only once? No, p can divide multiple times)
            # We need to divide by p as many times as it divides.
            # Since we are doing this for each prime, and primes are distinct,
            # we only need to divide by p as many times as it divides the number.
            # However, `remaining` holds the part not yet divided.
            # If we just do `remaining[idxs] //= p`, we only remove one factor of p.
            # We need to remove ALL factors of p.
            
            # Vectorized "divide by p while divisible":
            # This is hard to do in a single vectorized op without a loop.
            # But we can use a loop for the division part, or just accept that 
            # `remaining` might still have factors of p if we don't loop.
            # Actually, for the purpose of finding the LARGE prime factor at the end,
            # it doesn't matter if `remaining` still has small prime factors, 
            # AS LONG AS we don't mistake a small prime for the large one.
            # But we DO mistake it if `remaining` is not fully reduced.
            # Example: n = 12. p=2. remaining becomes 6 (if we only divide once).
            # Next p=3. remaining becomes 2.
            # End of primes. remaining is 2. 
            # If we check `remaining > 1`, we think 2 is a large prime factor.
            # But 2 was already processed.
            # So we MUST fully reduce `remaining` by each prime p.
            
            # We can do this with a small loop for the division, or 
            # we can just use the `phi` array to check if p still divides?
            # No, `phi` is modified.
            
            # Let's use a while loop inside the vectorized index selection? No, that's slow.
            # Alternative: 
            # Just do `remaining[idxs] //= p` and repeat until no more p divides?
            # We can check `remaining[idxs] % p == 0` again.
            
            # Actually, since p is small, the exponent is small.
            # Let's just do a few passes or a while loop on the numpy array?
            # Numpy doesn't have a built-in "while divisible".
            
            # Let's try a different approach for `remaining`:
            # We don't actually need `remaining` to be fully reduced by small primes 
            # if we are careful.
            # The large prime factor $q$ is the part of $n$ that is not divisible by any $p \le \sqrt{R}$.
            # If we divide `remaining` by $p$ only once, it might still be divisible by $p$.
            # But if we process ALL primes $p \le \sqrt{R}$, and for each $p$ we divide 
            # `remaining` by $p$ as many times as possible, then `remaining` will be 
            # coprime to all $p \le \sqrt{R}$.
            
            # To divide as many times as possible in NumPy:
            # We can use a loop that runs at most log_p(n) times.
            # For p=2, log2(10^12) ~ 40.
            # For large p, it's 1.
            
            # Let's optimize the division:
            while np.any(remaining[idxs] % p == 0):
                mask = remaining[idxs] % p == 0
                remaining[idxs][mask] //= p # This might be slow due to advanced indexing
            
            # Actually, `remaining[idxs]` is a copy? No, it's a view if idxs is a slice? 
            # `np.arange` creates a new array. `idxs` is an array of indices.
            # `remaining[idxs]` returns a copy. We need to modify the original.
            # So we must do:
            # vals = remaining[idxs]
            # while np.any(vals % p == 0):
            #     mask = vals % p == 0
            #     vals[mask] //= p
            # remaining[idxs] = vals
            
            # This is getting complicated.
            # Let's stick to the simpler approach: 
            # Just divide by p once. 
            # Wait, if we only divide by p once, `remaining` might still have p.
            # Then when we check `remaining > 1` at the end, we might identify p as the large factor.
            # But p is small. 
            # We can just check if `remaining` is prime? No.
            
            # Let's go back to the standard algorithm:
            # We can just compute `phi` correctly.
            # The `remaining` array is ONLY used to detect the large prime factor.
            # If we fail to remove all small factors, we might incorrectly identify a small prime as large.
            # To avoid this, we can just check: if `remaining[idx] > 1`, 
            # is it a prime? We can check if it's in our list of small primes? 
            # No, it's > sqrt(R), so it's not in the list.
            # But if we didn't remove all factors of 2, `remaining` could be 2.
            # 2 is not > sqrt(R). 
            # So if `remaining[idx] > 1` AND `remaining[idx] <= sqrt(R)`, it's a small prime we missed?
            # No, if we process all primes up to sqrt(R), we should have removed them.
            # Unless we didn't remove them ALL.
            
            # Let's ensure we remove all factors.
            # We'll use the "copy and update" method.
            
            vals = remaining[idxs]
            while np.any(vals % p == 0):
                mask = vals % p == 0
                vals[mask] //= p
            remaining[idxs] = vals

        # After loop, remaining[idx] is either 1 or a prime > sqrt(R)
        mask = remaining > 1
        if np.any(mask):
            phi[mask] -= phi[mask] // remaining[mask]
            
        return phi
