import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 130s limit, 20s buffer
    
    # Strategy: Enumerate M = k * sigma(k) for k up to some limit,
    # find the M with the most solutions k_i such that k_i * sigma(k_i) = M.
    # Key insight: we need k_i <= sqrt(M). Since M = k*sigma(k) ~ k^2 * log(log(k)),
    # sqrt(M) ~ k * sqrt(log(log(k))), so k_i <= sqrt(M) is automatically satisfied
    # for all solutions since sigma(k) >= k+1 means M >= k(k+1) > k^2, so k < sqrt(M).
    
    # We'll use a sieve to compute sigma(k) for k up to K_max.
    # Then compute M_k = k * sigma(k) for each k, and count occurrences.
    
    # The challenge: K_max needs to be large enough to find good collisions.
    # M <= 10^18, and M ~ k^2 * log(log(k)), so k can be up to ~10^9.
    # But sieving up to 10^9 is too slow/memory intensive.
    
    # Better approach: Use the fact that for "nice" M values (like highly composite numbers),
    # there may be many k with k*sigma(k) = M. Let's try a different strategy:
    
    # Strategy 2: Pick M to be a product of many small primes, then find k dividing M
    # such that k*sigma(k) = M. Actually this is hard.
    
    # Strategy 3: Use numpy sieve with a reasonable K_max, then find best M.
    # Let's try K_max = 10^7 or so, which should be fast.
    
    import numpy as np
    
    # Try multiple K_max values, starting with a smaller one to get a quick result
    # then trying larger ones if time permits
    
    def try_sieve(K_max):
        """Compute sigma via sieve and find best M."""
        sigma = np.zeros(K_max + 1, dtype=np.int64)
        
        # Sieve: for each d, add d to sigma[d], sigma[2d], sigma[3d], ...
        for d in range(1, K_max + 1):
            sigma[d::d] += d
        
        k_arr = np.arange(1, K_max + 1, dtype=np.int64)
        M_arr = k_arr * sigma[1:]
        
        # Filter M <= 10^18
        valid_mask = M_arr <= 10**18
        M_valid = M_arr[valid_mask]
        k_valid = k_arr[valid_mask]
        
        if len(M_valid) == 0:
            return None, []
        
        # Find M with most k values
        unique_M, inverse, counts = np.unique(M_valid, return_inverse=True, return_counts=True)
        
        max_idx = np.argmax(counts)
        max_M = unique_M[max_idx]
        max_count = counts[max_idx]
        
        k_solutions = k_valid[inverse == max_idx]
        k_list = k_solutions.tolist()
        
        # Verify: all k_i should satisfy k_i * sigma(k_i) = M and k_i <= sqrt(M)
        sqrt_M = math.isqrt(max_M)
        k_list = [k for k in k_list if k <= sqrt_M]
        
        return int(max_M), k_list
    
    # Start with a moderate K_max
    best_M = None
    best_k_list = []
    
    # Phase 1: Quick search with K_max = 5 million
    K_max_1 = 5_000_000
    if time.time() < deadline - 20:
        M1, k1 = try_sieve(K_max_1)
        if M1 is not None and len(k1) > len(best_k_list):
            best_M = M1
            best_k_list = k1
    
    # Phase 2: Try larger K_max if time permits
    K_max_2 = 20_000_000
    if time.time() < deadline - 30:
        M2, k2 = try_sieve(K_max_2)
        if M2 is not None and len(k2) > len(best_k_list):
            best_M = M2
            best_k_list = k2
    
    # Phase 3: Even larger if time permits
    K_max_3 = 50_000_000
    if time.time() < deadline - 40:
        M3, k3 = try_sieve(K_max_3)
        if M3 is not None and len(k3) > len(best_k_list):
            best_M = M3
            best_k_list = k3
    
    if best_M is None or not best_k_list:
        return {'M': 1, 'k': [1]}
    
    # Final verification
    # Verify each k: k * sigma(k) == M
    def sigma(n):
        """Compute sum of divisors of n."""
        if n == 1:
            return 1
        result = 1
        temp = n
        for p in range(2, int(math.isqrt(temp)) + 1):
            if temp % p == 0:
                factor_sum = 1
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    factor_sum += power
                result *= factor_sum
        if temp > 1:
            result *= (1 + temp)
        return result
    
    verified_k = []
    for k in best_k_list:
        if k * sigma(k) == best_M and k <= math.isqrt(best_M):
            verified_k.append(k)
    
    if not verified_k:
        return {'M': 1, 'k': [1]}
    
    return {'M': int(best_M), 'k': verified_k}
