import math
import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Known solution: M=333312 with k=[336, 372, 434]
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    # Use NumPy vectorized sieve to compute sigma(k) for all k up to N
    # This is O(N log N) but with very fast C-level operations
    N = 10000000  # 10 million
    
    # Allocate sigma array
    sigma_arr = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for each d, add d to all multiples of d
    for d in range(1, N + 1):
        sigma_arr[d::d] += d
        # Check time every 1000 iterations to avoid overhead
        if d % 1000 == 0 and time.time() > deadline - 20.0:
            N = d  # Reduce N to what we've completed
            break
    
    # Now group by M = k * sigma(k)
    # Use vectorized operations for speed
    ks = np.arange(1, N + 1, dtype=np.int64)
    M_vals = ks * sigma_arr[1:N+1]
    
    # Filter: M <= 10^18 and k <= sqrt(M)
    # k <= sqrt(M) means k^2 <= M, i.e., k^2 <= k*sigma(k), i.e., k <= sigma(k)
    valid_mask = (M_vals <= 10**18) & (ks <= sigma_arr[1:N+1])
    valid_ks = ks[valid_mask]
    valid_Ms = M_vals[valid_mask]
    
    # Group by M value using numpy unique with inverse
    unique_Ms, inverse = np.unique(valid_Ms, return_inverse=True)
    
    # Count occurrences of each M
    counts = np.bincount(inverse)
    
    # Find the M with the most solutions
    if len(counts) > 0:
        max_count = int(np.max(counts))
        if max_count > best_count:
            best_idx = int(np.argmax(counts))
            best_M = int(unique_Ms[best_idx])
            # Get all k values for this M
            best_k = valid_ks[inverse == best_idx].tolist()
            best_k = [int(x) for x in best_k]
            best_count = max_count
    
    # Secondary search: try to find additional solutions for the best M
    # by checking k values beyond N that might also satisfy k*sigma(k) = best_M
    # For k > N, we need k <= sqrt(best_M) and k*sigma(k) = best_M
    # This means sigma(k) = best_M / k, so k must divide best_M
    # and sigma(k) = best_M // k
    
    if best_count >= 3 and time.time() < deadline - 5.0:
        M_target = best_M
        # k must be a divisor of M_target and k <= sqrt(M_target)
        # and sigma(k) = M_target // k
        # Also k > N (we already checked k <= N)
        
        sqrt_M = math.isqrt(M_target)
        # Find all divisors of M_target up to sqrt_M
        # Factor M_target first
        temp = M_target
        factors = {}
        d = 2
        while d * d <= temp:
            while temp % d == 0:
                factors[d] = factors.get(d, 0) + 1
                temp //= d
            d += 1
        if temp > 1:
            factors[temp] = factors.get(temp, 0) + 1
        
        # Generate all divisors up to sqrt_M
        def gen_divisors(factors_dict):
            if not factors_dict:
                return [1]
            items = list(factors_dict.items())
            result = [1]
            for prime, exp in items:
                new_result = []
                for r in result:
                    for e in range(exp + 1):
                        new_result.append(r * prime**e)
                result = new_result
            return result
        
        all_divs = gen_divisors(factors)
        existing_ks = set(best_k)
        
        for k_candidate in all_divs:
            if k_candidate > N and k_candidate <= sqrt_M:
                # Check if sigma(k_candidate) == M_target // k_candidate
                sigma_needed = M_target // k_candidate
                # Compute sigma(k_candidate) by factorization
                temp_k = k_candidate
                sigma_val = 1
                d2 = 2
                while d2 * d2 <= temp_k:
                    if temp_k % d2 == 0:
                        cnt = 0
                        p_power = 1
                        while temp_k % d2 == 0:
                            temp_k //= d2
                            cnt += 1
                            p_power *= d2
                        sigma_val *= (p_power * d2 - 1) // (d2 - 1)
                    d2 += 1
                if temp_k > 1:
                    sigma_val *= (temp_k + 1)
                
                if sigma_val == sigma_needed:
                    best_k.append(int(k_candidate))
                    best_count += 1
        
        # Also check divisors <= N that we might have missed (shouldn't happen, but just in case)
        for k_candidate in all_divs:
            if k_candidate <= N and k_candidate not in existing_ks and k_candidate <= sqrt_M:
                sigma_needed = M_target // k_candidate
                temp_k = k_candidate
                sigma_val = 1
                d2 = 2
                while d2 * d2 <= temp_k:
                    if temp_k % d2 == 0:
                        cnt = 0
                        p_power = 1
                        while temp_k % d2 == 0:
                            temp_k //= d2
                            cnt += 1
                            p_power *= d2
                        sigma_val *= (p_power * d2 - 1) // (d2 - 1)
                    d2 += 1
                if temp_k > 1:
                    sigma_val *= (temp_k + 1)
                
                if sigma_val == sigma_needed:
                    best_k.append(int(k_candidate))
                    best_count += 1
    
    # Ensure we return valid types
    return {'M': int(best_M), 'k': [int(x) for x in best_k]}
