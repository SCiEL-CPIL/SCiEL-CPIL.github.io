import math
import time
import random

# Global prime sieve for small primes
def _get_primes(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    if limit >= 1:
        is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, val in enumerate(is_prime) if val]

PRIMES = _get_primes(10**6)

def construct():
    start_time = time.time()
    LIMIT = 110.0  # Safety margin for 120s limit
    MAX_S = 10**12
    MAX_K = 200
    
    # Fast phi for a single number using precomputed primes
    def phi_single(x):
        if x == 1:
            return 1
        result, m = x, x
        for p in PRIMES:
            if p * p > m:
                break
            if m % p == 0:
                result -= result // p
                while m % p == 0:
                    m //= p
        if m > 1:
            result -= result // m
        return result
    
    # Segmented phi computation for a window of consecutive integers
    # This is faster for 200 consecutive numbers than calling phi_single 200 times
    def phi_segment(start, length):
        """Compute phi for start, start+1, ..., start+length-1"""
        # Initialize: result[i] = start+i, remaining[i] = start+i
        result = [start + i for i in range(length)]
        remaining = [start + i for i in range(length)]
        
        # For each small prime, mark which numbers in the window are divisible
        for p in PRIMES:
            if p >= start + length:
                break
            # Find first multiple of p in [start, start+length-1]
            first = ((start + p - 1) // p) * p
            for j in range(first - start, length, p):
                if remaining[j] % p == 0:
                    result[j] -= result[j] // p
                    while remaining[j] % p == 0:
                        remaining[j] //= p
        
        # Handle remaining large prime factors
        for i in range(length):
            if remaining[i] > 1:
                result[i] -= result[i] // remaining[i]
        
        return result
    
    best_k = 1
    best_s = 1
    
    max_start = MAX_S - MAX_K + 1
    
    # Strategy: Use probabilistic pre-filtering
    # For large n, phi(n) collisions are rare. But we can identify "risky" windows
    # where collisions are more likely and skip them.
    
    # Key insight: phi(n) for random large n is roughly uniform in a range.
    # Collisions happen when two numbers have similar "smoothness" properties.
    # We can pre-filter by checking if the window contains numbers that are
    # "unusually smooth" (many small prime factors) which might cause phi values
    # to be close together.
    
    # Actually, let's just do efficient random sampling with the segmented phi.
    # The segmented approach is ~5-10x faster than individual phi calls.
    
    attempts = 0
    while time.time() - start_time < LIMIT:
        s = random.randint(1, max_start)
        attempts += 1
        
        # Compute phi for the entire window
        phis = phi_segment(s, MAX_K)
        
        # Check for distinctness
        if len(set(phis)) == MAX_K:
            return {'start': s, 'length': MAX_K}
        
        # If we've tried many times, try a deterministic scan as fallback
        if attempts % 100 == 0 and time.time() - start_time > LIMIT - 20:
            break
    
    # Fallback: deterministic scan from high end
    if time.time() - start_time < LIMIT - 5:
        current_end = MAX_S
        block_size = 5000  # Larger blocks for efficiency
        
        while current_end > 1 and time.time() - start_time < LIMIT - 5:
            block_start = max(1, current_end - block_size + 1)
            phis = phi_segment(block_start, block_size)
            
            # Find longest run of distinct values using sliding window
            seen_vals = {}
            l = 0
            max_len = 0
            max_start_idx = 0
            for r in range(len(phis)):
                if phis[r] in seen_vals and seen_vals[phis[r]] >= l:
                    l = seen_vals[phis[r]] + 1
                if r - l + 1 > max_len:
                    max_len = r - l + 1
                    max_start_idx = l
                seen_vals[phis[r]] = r
            
            if max_len > best_k:
                best_k = max_len
                best_s = block_start + max_start_idx
                if max_len >= MAX_K:
                    return {'start': best_s, 'length': best_k}
            
            current_end = block_start - 1
    
    return {'start': best_s, 'length': best_k}
