import math
import time
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s buffer for 150s limit
    
    # Strategy: Sieve sigma(k) for k up to N.
    # Find M with maximum number of k such that k*sigma(k) = M.
    # N=10^7 is a good target. Memory: 10^7 * 8 bytes = 80MB.
    # Time: Sieve is O(N log log N). Collision check is O(N).
    
    # Determine max N based on available time? 
    # Let's start with 10^7. If it's too slow, we might need to reduce, 
    # but 10^7 is generally fast in Python if using list operations or efficient loops.
    # However, pure Python loop for sieve is slow. 
    # We can use a linear sieve or standard sieve.
    # Standard sieve for sigma:
    # sigma[n] = 1
    # for i in range(2, N+1):
    #   for j in range(i, N+1, i):
    #     sigma[j] += i
    # This is O(N log N). For N=10^7, sum(1/i) ~ log(10^7) ~ 16. 
    # 10^7 * 16 = 1.6 * 10^8 operations. In Python, this might take ~10-20 seconds.
    # Acceptable.
    
    # We will increase N if time permits, but 10^7 is a safe bet for 5+ collisions.
    # Known 5-way collisions exist around M ~ 10^7 - 10^8.
    
    N = 10_000_000
    
    # Initialize sigma array
    # Using list of ints. 
    # To save memory and time, we can use array('L') or just list. List is faster for access in Python 3.
    sigma = [1] * (N + 1)
    
    # Sieve
    # Optimization: Only loop i up to sqrt(N) for the inner loop? No, sigma needs all divisors.
    # We must loop i from 2 to N.
    # But we can skip if i > N//2? No, i is a divisor.
    # Standard sieve:
    for i in range(2, N + 1):
        # If i is large, the inner loop runs few times.
        # To speed up, we can break early if i > N//2? No.
        # Actually, we can just run it.
        for j in range(i, N + 1, i):
            sigma[j] += i
            
    # Check time
    elapsed = time.time() - start_time
    if elapsed > 130:
        # Fallback if sieve is too slow
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Find collisions
    # Map M -> list of k
    # M = k * sigma[k]
    # M can be up to N * sigma(N). sigma(N) ~ N log log N. 
    # So M ~ 10^7 * 10^7 * 16 ~ 10^15. Fits in int.
    
    collisions = defaultdict(list)
    for k in range(1, N + 1):
        m_val = k * sigma[k]
        collisions[m_val].append(k)
        
    # Find the M with the most k's
    best_M = 0
    best_k_list = []
    max_count = 0
    
    for m_val, ks in collisions.items():
        if len(ks) > max_count:
            max_count = len(ks)
            best_M = m_val
            best_k_list = ks
            
            # Early exit if we find a very high count? 
            # Unlikely to jump from 5 to 10 quickly.
            
    # Verify the best solution
    # The task requires exact factorization check.
    # We must ensure k_i * sigma(k_i) == M for all k_i.
    # Our sieve computed sigma correctly, so this should hold.
    # But we must ensure k_i <= isqrt(M).
    # And k_i are distinct (they are, as keys in dict are unique per k in loop).
    
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    sqrt_M = math.isqrt(best_M)
    valid_ks = [k for k in best_k_list if k <= sqrt_M]
    
    # If valid_ks is empty, something is wrong (k*sigma(k)=M implies k <= sqrt(M) for k>1? 
    # If k=1, M=1. 
    # For k>1, sigma(k) >= k+1, so k*sigma(k) > k^2, so k < sqrt(M).
    # So all k should be valid.)
    
    if len(valid_ks) < 3:
        # Fallback to known 3-way
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Double check with exact factorization for a few samples to be safe?
    # The problem says "checked with exact factorization". 
    # Our sieve is exact for integers < 2^53. 
    # We should verify the top ones to be absolutely sure.
    
    def exact_sigma(n):
        if n == 1:
            return 1
        result = 1
        temp = n
        p = 2
        while p * p <= temp:
            if temp % p == 0:
                term = 1
                while temp % p == 0:
                    temp //= p
                    term *= p
                result *= (term * (p + 1) // (p - 1))
            p += 1
        if temp > 1:
            result *= (temp + 1)
        return result
        
    # Verify all valid_ks
    verified_ks = []
    for k in valid_ks:
        if k * exact_sigma(k) == best_M:
            verified_ks.append(k)
            
    if len(verified_ks) >= 3:
        return {'M': int(best_M), 'k': [int(k) for k in verified_ks]}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
