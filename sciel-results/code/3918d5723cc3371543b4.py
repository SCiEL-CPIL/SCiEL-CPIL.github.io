import math
import time

def construct():
    start_time = time.time()
    LIMIT = 115.0  # seconds (safety margin before 120s kill)
    MAX_S = 10**12

    # Generate primes up to 10^6 for trial division
    limit_p = 10**6
    is_prime = [True] * (limit_p + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit_p**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit_p + 1, i):
                is_prime[j] = False
    primes = [i for i, val in enumerate(is_prime) if val]
    
    # For segmented phi: we need primes up to sqrt(MAX_S) = 10^6
    # But for a window of size W, we only need primes up to W for the sieve part,
    # plus trial division for remaining factors.
    
    # Segmented phi computation for a window [start, start+W-1]
    def segmented_phi(start, W):
        """Compute phi(n) for n in [start, start+W-1] using segmented approach."""
        # Initialize arrays
        vals = list(range(start, start + W))  # remaining values to factor
        phi = list(range(start, start + W))   # phi values, start with n itself
        
        # For each prime p, mark multiples in the window
        # We need primes up to sqrt(start + W - 1) for full factorization
        # But for efficiency, we use primes up to W for the sieve,
        # then handle remaining factors via trial division
        
        # Actually, for a window of size W, we can use all primes up to W
        # to do the segmented sieve part
        max_p_for_sieve = min(W, limit_p)
        
        for p in primes:
            if p > max_p_for_sieve:
                break
            # Find first multiple of p in [start, start+W-1]
            first = ((start + p - 1) // p) * p
            if first < start:
                first += p
            # Mark all multiples
            for j in range(first, start + W, p):
                idx = j - start
                if vals[idx] % p == 0:
                    phi[idx] -= phi[idx] // p
                    while vals[idx] % p == 0:
                        vals[idx] //= p
        
        # For remaining values > 1, they are prime factors
        for i in range(W):
            if vals[i] > 1:
                phi[i] -= phi[i] // vals[i]
        
        return phi

    best_k = 1
    best_s = 1
    
    # Strategy: Scan windows of size 200 at strategic positions
    # Try near 10^12, then other large values
    # Window size 200 means we check if all 200 phi values are distinct
    
    WINDOW = 200
    
    # Try multiple starting positions
    # 1. Near 10^12
    # 2. Random-ish large positions using deterministic hash
    # 3. Smaller positions if time permits
    
    def check_window(s):
        """Check if phi(s)..phi(s+WINDOW-1) are all distinct."""
        if s < 1 or s + WINDOW - 1 > MAX_S:
            return False
        phis = segmented_phi(s, WINDOW)
        # Check for duplicates
        seen = set()
        for v in phis:
            if v in seen:
                return False
            seen.add(v)
        return True
    
    def find_best_in_block(s, block_size):
        """Find the longest run of distinct phi values in [s, s+block_size-1]."""
        if s < 1 or s + block_size - 1 > MAX_S:
            return (1, s)
        
        # Compute phis for the block
        phis = segmented_phi(s, block_size)
        
        # Find longest run of distinct values using two-pointer
        seen_vals = {}
        l = 0
        max_len = 0
        max_start_idx = 0
        
        for r in range(len(phis)):
            v = phis[r]
            if v in seen_vals:
                prev_idx = seen_vals[v]
                if prev_idx >= l:
                    l = prev_idx + 1
            seen_vals[v] = r
            
            curr_len = r - l + 1
            if curr_len > max_len:
                max_len = curr_len
                max_start_idx = l
                if max_len >= 200:
                    break
        
        return (max_len, s + max_start_idx)
    
    # Search strategy:
    # Phase 1: Try windows of size 200 at various positions near 10^12
    # Phase 2: Scan larger blocks to find good candidates
    # Phase 3: If time permits, explore more
    
    # Phase 1: Check if any 200-window near 10^12 works
    # Try a few positions
    test_positions = []
    # Near 10^12
    for offset in [0, 1000, 5000, 10000, 50000, 100000]:
        pos = MAX_S - WINDOW - offset
        if pos >= 1:
            test_positions.append(pos)
    
    # Deterministic pseudo-random positions
    import hashlib
    for i in range(50):
        h = int(hashlib.md5(f"pos_{i}".encode()).hexdigest(), 16)
        pos = 1 + h % (MAX_S - WINDOW)
        test_positions.append(pos)
    
    for s in test_positions:
        if time.time() - start_time > LIMIT - 10:
            break
        if check_window(s):
            return {'start': s, 'length': 200}
    
    # Phase 2: Scan blocks to find best run
    # Use larger blocks for better coverage
    block_size = 5000
    # Scan from high to low, but skip some regions to save time
    scan_regions = [
        (MAX_S - block_size, MAX_S),
        (MAX_S - 10**7 - block_size, MAX_S - 10**7),
        (MAX_S - 10**8 - block_size, MAX_S - 10**8),
        (MAX_S - 10**9 - block_size, MAX_S - 10**9),
        (10**6, 10**6 + block_size),
        (10**8, 10**8 + block_size),
    ]
    
    for region_start, region_end in scan_regions:
        if time.time() - start_time > LIMIT - 5:
            break
        if region_start < 1 or region_end > MAX_S:
            continue
        # Scan this region
        cur = region_end
        while cur >= region_start:
            if time.time() - start_time > LIMIT - 5:
                break
            block_s = max(region_start, cur - block_size + 1)
            if block_s < 1:
                block_s = 1
            k, s = find_best_in_block(block_s, min(block_size, cur - block_s + 1))
            if k > best_k:
                best_k = k
                best_s = s
                if best_k >= 200:
                    return {'start': best_s, 'length': 200}
            cur = block_s - 1
    
    return {'start': best_s, 'length': best_k}
