import time
import math
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin

    def now():
        return time.time()

    # 1. Precompute primes up to 10^6 using numpy sieve
    LIMIT = 10**6
    is_prime = np.ones(LIMIT + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(LIMIT)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = [int(p) for p in np.where(is_prime)[0] if p >= 2]

    # 2. Segmented phi computation for range [lo, hi]
    def segmented_phi(lo, hi):
        n = hi - lo + 1
        if n <= 0:
            return np.array([], dtype=np.int64)
        
        # Initialize: phi[i] = lo + i
        phi_vals = np.arange(lo, hi + 1, dtype=np.int64)
        
        # For each prime p, process multiples in [lo, hi]
        for p in primes:
            if p * p > hi:
                break
            
            # Find first multiple of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            
            if start > hi:
                continue
            
            # Indices where original number is multiple of p
            indices = np.arange(start - lo, n, p)
            
            if len(indices) > 0:
                phi_vals[indices] = phi_vals[indices] - phi_vals[indices] // p
        
        return phi_vals

    # 3. Find longest run of distinct values in array using sliding window
    def max_distinct_run(vals):
        """vals is a numpy array. Returns (best_k, best_start_idx)."""
        n = len(vals)
        if n == 0:
            return 0, 0
        
        best_k = 1
        best_start = 0
        
        # Sliding window with hash map
        left = 0
        count_map = {}
        
        for right in range(n):
            v = int(vals[right])
            count_map[v] = count_map.get(v, 0) + 1
            
            # Shrink from left until all distinct
            while count_map[v] > 1:
                lv = int(vals[left])
                count_map[lv] -= 1
                left += 1
            
            curr_k = right - left + 1
            if curr_k > best_k:
                best_k = curr_k
                best_start = left
            
            if best_k == 200:
                break
        
        return best_k, best_start

    best_k = 1
    best_s = 1

    # 4. Search strategy: try large contiguous ranges
    # Focus on ranges where we might find long runs of distinct phi values
    
    # Try ranges at various scales
    ranges_to_try = []
    
    # Try windows of size 10^6 at various positions
    for base in [10**6, 10**7, 10**8, 10**9, 10**10, 10**11]:
        lo = base
        hi = min(base + 10**6, 10**12)
        if lo >= 1 and hi >= lo:
            ranges_to_try.append((lo, hi))
    
    # Also try some random ranges
    random.seed(42)
    for _ in range(15):
        lo = random.randint(10**6, 10**12 - 10**6)
        hi = lo + 10**6
        if hi <= 10**12:
            ranges_to_try.append((lo, hi))
    
    # Process ranges
    for lo, hi in ranges_to_try:
        if now() > DEADLINE - 5.0:
            break
        
        if lo < 1:
            lo = 1
        if hi > 10**12:
            hi = 10**12
        if lo > hi:
            continue
        
        try:
            phi_vals = segmented_phi(lo, hi)
            k, idx = max_distinct_run(phi_vals)
            
            if k > best_k:
                best_k = k
                best_s = lo + idx
            
            if best_k == 200:
                break
                
        except Exception:
            continue
    
    return {'start': best_s, 'length': best_k}
