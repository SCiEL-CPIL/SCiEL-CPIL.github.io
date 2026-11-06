import math
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0
    
    def now():
        return time.time()
    
    def phi_segmented_numpy(L, R):
        """
        Compute phi(n) for n in [L, R] using NumPy vectorization.
        Returns a numpy array of phi values.
        """
        size = R - L + 1
        if size <= 0:
            return np.array([], dtype=np.int64)
        
        # num[i] holds the remaining part of (L+i) after dividing by small primes
        # phi[i] holds the current value of phi(L+i)
        num = np.arange(L, R + 1, dtype=np.int64)
        phi = num.copy()
        
        # Sieve primes up to sqrt(R)
        limit = int(math.isqrt(R)) + 1
        
        # Simple sieve for small primes
        is_prime = np.ones(limit + 1, dtype=bool)
        if limit >= 0:
            is_prime[0] = False
        if limit >= 1:
            is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                is_prime[i*i::i] = False
        primes = np.where(is_prime)[0]
        
        for p in primes:
            # Find the first multiple of p in [L, R]
            start = ((L + p - 1) // p) * p
            # Get indices of multiples of p in [L, R]
            indices = np.arange(start, R + 1, p) - L
            
            # Update phi: phi[n] -= phi[n] // p
            phi[indices] -= phi[indices] // p
            
            # Remove all factors of p from num
            # We need to divide num[indices] by p until not divisible
            # This is tricky with NumPy, so we'll do it in a loop for each prime
            # But we can vectorize the check
            mask = num[indices] % p == 0
            while mask.any():
                num[indices[mask]] //= p
                mask = num[indices] % p == 0
        
        # Finalize: if num[i] > 1, it is a prime factor
        remaining = num > 1
        if remaining.any():
            phi[remaining] -= phi[remaining] // num[remaining]
            
        return phi
    
    best_start = 1
    best_len = 1
    
    def update_best(s, k):
        nonlocal best_start, best_len
        if k > best_len:
            best_len = k
            best_start = s
    
    def find_longest_distinct_run(phi_vals, offset):
        """
        Find the longest run of consecutive distinct values in phi_vals.
        offset is the starting index (1-based) for reporting.
        Returns (best_k, best_s) where best_s is 1-based.
        """
        n = len(phi_vals)
        if n == 0:
            return 0, 0
        
        window_start = 0
        seen_set = set()
        max_k = 0
        max_s = 0
        
        for i in range(n):
            v = phi_vals[i]
            # If v is already in the set, we need to remove elements from the left
            # until v is no longer in the set.
            while v in seen_set:
                seen_set.remove(phi_vals[window_start])
                window_start += 1
            
            seen_set.add(v)
            current_len = i - window_start + 1
            if current_len > max_k:
                max_k = current_len
                max_s = window_start + offset  # 1-based index
        
        return max_k, max_s
    
    # Search strategy:
    # 1. First search [1, 10^7] in one big block
    # 2. Then search larger ranges if time permits
    
    # Phase 1: Search [1, 10^7]
    L1 = 1
    R1 = 10**7
    if now() < deadline:
        phi_vals1 = phi_segmented_numpy(L1, R1)
        k1, s1 = find_longest_distinct_run(phi_vals1, L1)
        update_best(s1, k1)
        print(f"Phase 1: Found k={k1} at s={s1}")
    
    # Phase 2: Search [10^7, 2*10^7] with overlap
    if now() < deadline - 30:
        L2 = 10**7 - 1000
        R2 = 2 * 10**7
        phi_vals2 = phi_segmented_numpy(L2, R2)
        k2, s2 = find_longest_distinct_run(phi_vals2, L2)
        update_best(s2, k2)
        print(f"Phase 2: Found k={k2} at s={s2}")
    
    # Phase 3: Search [2*10^7, 3*10^7] with overlap
    if now() < deadline - 20:
        L3 = 2 * 10**7 - 1000
        R3 = 3 * 10**7
        phi_vals3 = phi_segmented_numpy(L3, R3)
        k3, s3 = find_longest_distinct_run(phi_vals3, L3)
        update_best(s3, k3)
        print(f"Phase 3: Found k={k3} at s={s3}")
    
    # Phase 4: Search [3*10^7, 4*10^7] with overlap
    if now() < deadline - 15:
        L4 = 3 * 10**7 - 1000
        R4 = 4 * 10**7
        phi_vals4 = phi_segmented_numpy(L4, R4)
        k4, s4 = find_longest_distinct_run(phi_vals4, L4)
        update_best(s4, k4)
        print(f"Phase 4: Found k={k4} at s={s4}")
    
    # Phase 5: Search [4*10^7, 5*10^7] with overlap
    if now() < deadline - 10:
        L5 = 4 * 10**7 - 1000
        R5 = 5 * 10**7
        phi_vals5 = phi_segmented_numpy(L5, R5)
        k5, s5 = find_longest_distinct_run(phi_vals5, L5)
        update_best(s5, k5)
        print(f"Phase 5: Found k={k5} at s={s5}")
    
    # Phase 6: Search [5*10^7, 6*10^7] with overlap
    if now() < deadline - 5:
        L6 = 5 * 10**7 - 1000
        R6 = 6 * 10**7
        phi_vals6 = phi_segmented_numpy(L6, R6)
        k6, s6 = find_longest_distinct_run(phi_vals6, L6)
        update_best(s6, k6)
        print(f"Phase 6: Found k={k6} at s={s6}")
    
    # Phase 7: Search [6*10^7, 7*10^7] with overlap
    if now() < deadline - 3:
        L7 = 6 * 10**7 - 1000
        R7 = 7 * 10**7
        phi_vals7 = phi_segmented_numpy(L7, R7)
        k7, s7 = find_longest_distinct_run(phi_vals7, L7)
        update_best(s7, k7)
        print(f"Phase 7: Found k={k7} at s={s7}")
    
    # Phase 8: Search [7*10^7, 8*10^7] with overlap
    if now() < deadline - 2:
        L8 = 7 * 10**7 - 1000
        R8 = 8 * 10**7
        phi_vals8 = phi_segmented_numpy(L8, R8)
        k8, s8 = find_longest_distinct_run(phi_vals8, L8)
        update_best(s8, k8)
        print(f"Phase 8: Found k={k8} at s={s8}")
    
    # Phase 9: Search [8*10^7, 9*10^7] with overlap
    if now() < deadline - 1:
        L9 = 8 * 10**7 - 1000
        R9 = 9 * 10**7
        phi_vals9 = phi_segmented_numpy(L9, R9)
        k9, s9 = find_longest_distinct_run(phi_vals9, L9)
        update_best(s9, k9)
        print(f"Phase 9: Found k={k9} at s={s9}")
    
    # Phase 10: Search [9*10^7, 10^8] with overlap
    if now() < deadline - 0.5:
        L10 = 9 * 10**7 - 1000
        R10 = 10**8
        phi_vals10 = phi_segmented_numpy(L10, R10)
        k10, s10 = find_longest_distinct_run(phi_vals10, L10)
        update_best(s10, k10)
        print(f"Phase 10: Found k={k10} at s={s10}")
    
    print(f"Final best: k={best_len} at s={best_start}")
    
    return {'start': best_start, 'length': best_len}
