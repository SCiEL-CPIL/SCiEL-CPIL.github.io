import time
import sys
import numpy as np
from math import isqrt

def _generate_powerful_numbers(limit):
    """
    Generate all powerful numbers up to limit.
    Powerful number: n = a^2 * b^3 where b is square-free.
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
    deadline = start_time + 115  # Safety margin
    
    best_N = 5
    best_pairs = [[1, 4]]
    best_count = 1
    
    limits_to_try = [2_000_000_000, 1_000_000_000, 500_000_000, 200_000_000]
    
    for limit in limits_to_try:
        try:
            if time.time() > deadline - 25:
                break
                
            gen_start = time.time()
            powerful = _generate_powerful_numbers(limit)
            gen_time = time.time() - gen_start
            
            if time.time() > deadline - 15:
                continue
                
            K = len(powerful)
            if K == 0:
                continue
            
            # If K is too large, the all_sums array will be too big
            # K^2/2 elements, each 8 bytes. For K=30000, that's 3.6GB.
            if K > 35000:
                continue
                
            pow_arr = np.array(powerful, dtype=np.int64)
            
            # Collect all sums using vectorized approach
            all_sums = []
            for i in range(K):
                if time.time() > deadline - 10:
                    break
                all_sums.append(pow_arr[i] + pow_arr[i:])
            
            if not all_sums:
                continue
            
            all_sums = np.concatenate(all_sums)
            del all_sums  # Free the list of arrays
            
            if time.time() > deadline - 5:
                continue
            
            unique, cnt = np.unique(all_sums, return_counts=True)
            del all_sums
            
            if time.time() > deadline - 2:
                continue
            
            # Find the N with max count
            best_idx = np.argmax(cnt)
            current_best_N = int(unique[best_idx])
            current_best_count = int(cnt[best_idx])
            
            if current_best_count > best_count:
                best_count = current_best_count
                best_N = current_best_N
                # Reconstruct pairs
                best_pairs = []
                pow_set = set(powerful)
                half_N = best_N // 2
                for a in powerful:
                    if a > half_N:
                        break
                    b = best_N - a
                    if b in pow_set:
                        best_pairs.append([a, b])
                    
        except MemoryError:
            continue
        except Exception:
            continue
            
    return {'N': best_N, 'pairs': best_pairs}

if __name__ == "__main__":
    res = construct()
    print(res['N'], len(res['pairs']))
