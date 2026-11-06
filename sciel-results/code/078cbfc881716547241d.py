import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
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
    deadline = start_time + 115
    
    best_N = 5
    best_pairs = [[1, 4]]
    best_count = 1
    
    # Try increasing limits
    # K ~ sqrt(L)
    # NumPy outer sum KxK.
    # K=10000 -> 10^8 elements. ~800MB.
    # K=20000 -> 4*10^8 elements. ~3.2GB. Might fail.
    # So limit K to ~15000.
    # L ~ K^2 / c^2.
    
    limits_to_try = [10**7, 10**8, 3*10**8, 5*10**8]
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 30:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 20:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
            
            # If K is too large, skip to avoid memory/time issues
            if K > 20000:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Compute all sums a+b with a <= b
            # Using outer product and masking
            # To save memory, we can compute in chunks or just use the full matrix if K is small enough
            # For K=10000, 10^8 int64 is 800MB.
            # For K=20000, 4*10^8 int64 is 3.2GB.
            
            if K > 15000:
                # Fallback to dict counting for slightly larger K if memory fails?
                # Or just skip.
                continue

            # Create upper triangular mask
            # This is KxK booleans.
            # For K=15000, 2.25 * 10^8 booleans = 225 MB.
            
            # More efficient:
            # sums = np.add.outer(pow_arr, pow_arr)
            # This creates full KxK.
            # We only need upper triangle.
            
            # Let's try to do it in one go if K <= 10000
            if K <= 10000:
                sums = np.add.outer(pow_arr, pow_arr)
                # Mask for a <= b (upper triangle including diagonal)
                # np.triu_indices
                iu = np.triu_indices(K)
                sums = sums[iu]
                
                unique_sums, counts = np.unique(sums, return_counts=True)
                
                if len(counts) > 0:
                    max_idx = np.argmax(counts)
                    current_best_N = int(unique_sums[max_idx])
                    current_best_count = int(counts[max_idx])
                    
                    if current_best_count > best_count:
                        best_count = current_best_count
                        best_N = current_best_N
                        # Reconstruct pairs
                        best_pairs = []
                        pow_set = set(powerful)
                        for a in powerful:
                            b = best_N - a
                            if b < a:
                                continue
                            if b in pow_set:
                                best_pairs.append([a, b])
                        
            else:
                # For K > 10000, use dict counting (slower but less peak memory if we don't store matrix)
                # Actually dict counting is O(K^2) time and O(unique_sums) memory.
                # For K=15000, K^2 = 2.25 * 10^8.
                # Python loop is slow.
                # Let's just skip K > 10000 to be safe on time.
                pass
                
        except MemoryError:
            continue
        except Exception as e:
            continue
            
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
