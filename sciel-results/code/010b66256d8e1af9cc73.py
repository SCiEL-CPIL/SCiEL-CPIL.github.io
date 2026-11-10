"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import deque

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # safety margin before 120s kill

def construct():
    import math
    
    def sieve_primes(limit):
        """Generate all primes up to limit using sieve of Eratosthenes."""
        if limit < 2:
            return []
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                step = i
                start = i * i
                is_prime[start:limit+1:step] = bytearray(len(is_prime[start:limit+1:step]))
        return [i for i in range(2, limit+1) if is_prime[i]]
    
    # Precompute primes up to 10^6 (sufficient for trial division of numbers up to ~10^12)
    PRIME_LIMIT = 1000000
    primes = sieve_primes(PRIME_LIMIT)
    
    def phi_fast(x):
        """Compute Euler's totient using precomputed primes for trial division."""
        if x == 1:
            return 1
        result = x
        y = x
        for p in primes:
            if p * p > y:
                break
            if y % p == 0:
                result -= result // p
                while y % p == 0:
                    y //= p
        if y > 1:
            result -= result // y
        return result
    
    best_k = 1
    best_s = 1
    deadline = _START_TIME + _TIME_LIMIT - 5.0  # 5s safety margin
    
    MAX_K = 200
    
    def search_in_range(start, end, max_k):
        """Search for the longest run of distinct phi values in [start, end]."""
        n = end - start + 1
        if n <= 0:
            return 1, start
        
        # Compute phi values for the entire range
        phi_vals = [0] * n
        for i in range(n):
            if time.time() > deadline:
                return 1, start
            phi_vals[i] = phi_fast(start + i)
        
        # Use sliding window with deque to find longest run of distinct values
        best_local_k = 1
        best_local_s = start
        
        # Single pass with sliding window
        seen = {}  # value -> index
        left = 0
        
        for right in range(n):
            if time.time() > deadline:
                break
            
            val = phi_vals[right]
            
            # If val is already in the window, move left past its previous occurrence
            if val in seen:
                left = max(left, seen[val] + 1)
            
            seen[val] = right
            
            # Current window [left, right] has all distinct values
            current_k = right - left + 1
            
            if current_k > best_local_k:
                best_local_k = current_k
                best_local_s = start + left
            
            # If we've hit max_k, we can stop early
            if best_local_k >= max_k:
                break
        
        return best_local_k, best_local_s
    
    # Search strategy: focus on ranges where longer runs are more likely
    # Larger numbers tend to have more spread-out totient values
    # We'll search in strategic ranges across the number line
    
    ranges_to_try = [
        # Small ranges for baseline
        (1, 50000),
        # Medium ranges
        (1000000, 1050000),
        (10000000, 10050000),
        # Larger ranges where totients are more spread out
        (100000000, 100050000),
        (1000000000, 1000050000),
        (10000000000, 10000050000),
        (100000000000, 100000050000),
        # Near the upper bound
        (999999000000, 10**12),
    ]
    
    for start, end in ranges_to_try:
        if time.time() > deadline:
            break
        if end > 10**12:
            end = 10**12
        if start < 1:
            start = 1
        if start > end:
            continue
        
        k_found, s_found = search_in_range(start, end, MAX_K)
        if k_found > best_k:
            best_k = k_found
            best_s = s_found
        
        # If we found a reasonably long run, try to extend it
        if k_found >= 5:
            # Try extending left and right
            left_start = max(1, s_found - 500)
            right_end = min(10**12, s_found + k_found + 500)
            extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
            if extended_k > best_k:
                best_k = extended_k
                best_s = extended_s
    
    # If time remains, do additional targeted searches
    if time.time() < deadline - 5:
        # Try random sampling in large ranges
        import random
        random.seed(42)
        
        for _ in range(20):
            if time.time() > deadline:
                break
            
            # Pick a random start in a large range
            base = random.randint(10**6, 10**12 - 100000)
            s = base
            e = min(10**12, base + 50000)
            
            k_found, s_found = search_in_range(s, e, MAX_K)
            if k_found > best_k:
                best_k = k_found
                best_s = s_found
            
            # If promising, extend
            if k_found >= 5:
                left_start = max(1, s_found - 500)
                right_end = min(10**12, s_found + k_found + 500)
                extended_k, extended_s = search_in_range(left_start, right_end, MAX_K)
                if extended_k > best_k:
                    best_k = extended_k
                    best_s = extended_s
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
