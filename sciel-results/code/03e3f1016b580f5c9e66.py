import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is square-free.
    """
    if limit < 1:
        return []
    
    powerful = set()
    powerful.add(1)
    
    # b must be square-free and b^3 <= limit
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    if max_b < 1:
        max_b = 1

    # Check square-freeness using primes up to isqrt(max_b)
    sieve_limit = isqrt(max_b) + 1
    if sieve_limit < 2:
        primes = []
    else:
        is_prime = bytearray(b'\x01') * (sieve_limit + 1)
        is_prime[0] = 0
        if sieve_limit >= 1:
            is_prime[1] = 0
        for i in range(2, isqrt(sieve_limit) + 1):
            if is_prime[i]:
                for j in range(i * i, sieve_limit + 1, i):
                    is_prime[j] = 0
        primes = [i for i in range(2, sieve_limit + 1) if is_prime[i]]
        
    for b in range(1, max_b + 1):
        t = b
        is_sf = True
        for p in primes:
            pp = p * p
            if pp > t:
                break
            if t % pp == 0:
                is_sf = False
                break
        if not is_sf:
            continue
            
        b3 = b * b * b
        if b3 > limit:
            continue
            
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
                
    return sorted(powerful)

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    # Strategy: Generate powerful numbers up to a large limit, then use
    # vectorized binary search to count representations for candidate N values
    # in high-density regions, avoiding O(K^2) pair enumeration.
    
    best_result = {'N': 5, 'pairs': [[1, 4]]}
    
    # Try multiple limits; larger limit = more powerful numbers = potentially more pairs
    limits_to_try = [200_000_000_000, 100_000_000_000, 50_000_000_000, 20_000_000_000]
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 60:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 40:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            pow_set = set(powerful)
            
            # Identify high-density sum regions by sampling candidate N values
            # We'll look at sums where many pairs might exist
            # Use a sliding window approach: for each powerful number a, 
            # count how many b satisfy a+b = N for various N
            
            # Candidate N values: sample around midpoints and high-density areas
            # The maximum possible sum is 2*limit, minimum is 2
            # Focus on N values where the interval [N - max_pow, N - min_pow] 
            # overlaps significantly with [min_pow, max_pow]
            
            min_pow = int(pow_arr[0])
            max_pow = int(pow_arr[-1])
            
            # Generate candidate N values in the range [2*min_pow, 2*max_pow]
            # Sample strategically: more samples in the middle where density is higher
            num_candidates = 2000
            # Use non-uniform sampling: more candidates near the center
            # Center is at (min_pow + max_pow)
            center = (min_pow + max_pow) // 2
            spread = max_pow - min_pow
            
            # Create candidate N values with higher density near center
            # Use a combination of uniform and centered sampling
            candidates = []
            
            # Uniform sampling across the full range
            uniform_candidates = np.linspace(2 * min_pow, 2 * max_pow, num_candidates // 2, dtype=np.int64)
            candidates.extend(uniform_candidates.tolist())
            
            # Centered sampling (more density near center)
            centered_candidates = center + np.random.normal(0, spread // 3, num_candidates // 2).astype(np.int64)
            centered_candidates = np.clip(centered_candidates, 2 * min_pow, 2 * max_pow)
            candidates.extend(centered_candidates.tolist())
            
            # Also add some specific strategic values
            candidates.append(2 * center)
            candidates.append(center + spread // 4)
            candidates.append(center - spread // 4)
            
            # Remove duplicates and sort
            candidates = sorted(set(candidates))
            
            # For each candidate N, count representations using vectorized binary search
            best_N = None
            best_count = 0
            best_pairs = []
            
            # Process candidates in batches for efficiency
            batch_size = 100
            for batch_start in range(0, len(candidates), batch_size):
                if time.time() > deadline - 20:
                    break
                    
                batch = candidates[batch_start:batch_start + batch_size]
                
                for N in batch:
                    if N < 2 * min_pow or N > 2 * max_pow:
                        continue
                    
                    # Count pairs (a, b) with a <= b, a + b = N, a,b in powerful
                    # a ranges from min_pow to N - min_pow, and b = N - a
                    # We need a <= b, so a <= N/2
                    a_max = min(max_pow, N // 2)
                    a_min = max(min_pow, N - max_pow)
                    
                    if a_min > a_max:
                        continue
                    
                    # Binary search for each a in [a_min, a_max] to check if N-a is in powerful
                    # Vectorize: create array of a values, compute b = N - a, check membership
                    # But K can be large, so we need to be smart
                    
                    # Use numpy searchsorted to find which a values have N-a in powerful
                    # Create range of a values
                    a_values = np.arange(a_min, a_max + 1, dtype=np.int64)
                    b_values = N - a_values
                    
                    # Check which b_values are in pow_arr using searchsorted
                    # For each b, find its position in pow_arr and verify it exists
                    idx = np.searchsorted(pow_arr, b_values, side='left')
                    
                    # Check bounds and verify
                    valid_mask = (idx < K) & (pow_arr[idx] == b_values)
                    
                    count = int(np.sum(valid_mask))
                    
                    if count > best_count:
                        best_count = count
                        best_N = N
                        # Extract the actual pairs
                        valid_a = a_values[valid_mask]
                        valid_b = b_values[valid_mask]
                        best_pairs = [[int(a), int(b)] for a, b in zip(valid_a, valid_b)]
            
            if best_N is not None and best_count > 0:
                if best_N > 10**18:
                    continue
                if best_count > len(best_result['pairs']):
                    best_result = {'N': best_N, 'pairs': best_pairs}
                    
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return best_result

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
