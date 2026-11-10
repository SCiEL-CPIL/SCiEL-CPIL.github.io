def construct():
        import time
        start = time.time()
        deadline = start + 110.0
        
        N = 10_000_000 # 10 million
        
        # Sieve for sigma
        # Using list of ints
        sigma = [0] * (N + 1)
        
        # Optimization: Local variable lookup
        sig = sigma
        for i in range(1, N + 1):
            for j in range(i, N + 1, i):
                sig[j] += i
        
        # Now find collisions
        m_to_k = {}
        best_M = 333312
        best_k = [336, 372, 434]
        best_count = 3
        
        # We can also just iterate and update
        # To save memory, we only store k's for M's that have collided?
        # No, we need to know if it's a collision first.
        # So we store all M -> list?
        # Memory: 10^7 entries. Dict overhead is huge (~100 bytes/entry?). 1GB+.
        # This might OOM.
        
        # Alternative:
        # Compute M for all k, store in array?
        # M values are unique mostly.
        # We can sort (M, k) pairs?
        # Create list of tuples (M, k)?
        # 10^7 tuples of 2 ints. ~ 10^7 * 56 bytes = 560MB.
        # Plus the list overhead.
        # This is risky but might fit in 2GB/4GB RAM.
        
        # Let's try to be smarter.
        # We know most M are unique.
        # We can use a "bloom filter" or just a set of seen M?
        # If we use a set `seen`, and if M in seen, we need to retrieve the previous k's.
        # So we need a map M -> list of k's.
        # But we only add to the map if we see a duplicate?
        # No, we need the first k to compare.
        # So we must store M -> k for all unique M.
        
        # Is there a way to avoid storing all?
        # If we sort the M values, we can find adjacent duplicates.
        # List of M values: `M_vals = [k * sigma[k] for k in range(1, N+1)]`
        # This is 10^7 ints. ~ 80MB.
        # Then we need to map back to k?
        # We can create indices: `indices = list(range(1, N+1))`
        # Sort indices by M_vals[i]?
        # This requires storing M_vals and indices.
        # Then iterate through sorted indices.
        # If M_vals[indices[i]] == M_vals[indices[i-1]], we have a collision.
        # We can collect these k's.
        
        # This avoids the huge Dict overhead.
        # Memory:
        # sigma: 10^7 ints (80MB)
        # M_vals: 10^7 ints (80MB)
        # indices: 10^7 ints (80MB)
        # Total ~ 240MB + overhead. Safe.
        
        # Let's implement this.
        
        # 1. Compute sigma
        # 2. Compute M_vals
        # 3. Sort indices by M_vals
        # 4. Scan sorted indices for groups of same M
        # 5. Track max group size.
        
        # Check time after sigma calc?
        # If sigma calc takes too long, we return seed.
        
        if time.time() > deadline - 20:
            return {'M': best_M, 'k': best_k}
        
        # Compute M_vals
        M_vals = [0] * (N + 1)
        for k in range(1, N + 1):
            M_vals[k] = k * sig[k]
        
        if time.time() > deadline - 10:
            return {'M': best_M, 'k': best_k}
        
        # Sort indices
        indices = list(range(1, N + 1))
        indices.sort(key=lambda x: M_vals[x])
        
        # Scan
        current_M = M_vals[indices[0]]
        current_group = [indices[0]]
        
        for i in range(1, N):
            if M_vals[indices[i]] == current_M:
                current_group.append(indices[i])
            else:
                if len(current_group) > best_count:
                    best_count = len(current_group)
                    best_M = current_M
                    best_k = current_group[:]
                current_M = M_vals[indices[i]]
                current_group = [indices[i]]
                # Time check
                if time.time() > deadline - 5:
                    break
        
        # Check last group
        if len(current_group) > best_count:
            best_count = len(current_group)
            best_M = current_M
            best_k = current_group[:]
            
        return {'M': best_M, 'k': best_k}
