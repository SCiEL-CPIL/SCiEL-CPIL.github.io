import sys
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    
    # 1. Generate primes
    is_prime = [True] * (LIMIT + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(LIMIT**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, LIMIT + 1, i):
                is_prime[j] = False
            
    primes = [i for i, prime in enumerate(is_prime) if prime]
    prime_set = set(primes)
    
    # 2. Precompute next_prime_ge[x]
    # next_prime_ge[x] = smallest prime >= x
    next_prime_ge = [0] * (LIMIT + 2)
    next_p = None
    for i in range(LIMIT, -1, -1):
        if is_prime[i]:
            next_p = i
        if next_p is not None:
            next_prime_ge[i] = next_p
        else:
            next_prime_ge[i] = -1 # No prime >= i
    
    # 3. DP function
    # To avoid recursion depth issues and overhead, we can use iterative or just lru_cache
    # Depth is small, so recursion is fine.
    
    @lru_cache(maxsize=None)
    def calc_len(p, g):
        # Returns max length of sequence starting with p, last gap g
        # Next prime must be >= p + g
        target = p + g
        if target > LIMIT:
            return 1
        
        p_next = next_prime_ge[target]
        if p_next == -1 or p_next > LIMIT:
            return 1
            
        g_next = p_next - p
        return 1 + calc_len(p_next, g_next)

    best_len = 0
    best_p1 = -1
    best_p2 = -1
    
    # 4. Iterate over first two primes
    n = len(primes)
    for i in range(n):
        p1 = primes[i]
        for j in range(i + 1, n):
            p2 = primes[j]
            g1 = p2 - p1
            # Length is 1 (for p1) + calc_len(p2, g1)
            # Wait, calc_len(p, g) returns length of sequence *including* p?
            # My definition: "length of sequence starting with p".
            # So if seq is [p, p_next, ...], length is 1 + ...
            # Yes.
            # So total length = 1 + calc_len(p2, g1)
            
            # Optimization: if current potential max is less than best_len, skip?
            # Hard to know potential max without calculating.
            
            l = 1 + calc_len(p2, g1)
            if l > best_len:
                best_len = l
                best_p1 = p1
                best_p2 = p2
                
            # Time check
            if time.time() - start_time > 50:
                break
        if time.time() - start_time > 50:
            break
            
    if best_p1 == -1:
        # Fallback: return first prime or empty
        return [primes[0]] if primes else []

    # 5. Reconstruct sequence
    seq = [best_p1, best_p2]
    p = best_p2
    g = best_p2 - best_p1
    
    while True:
        target = p + g
        if target > LIMIT:
            break
        p_next = next_prime_ge[target]
        if p_next == -1 or p_next > LIMIT:
            break
        g_next = p_next - p
        seq.append(p_next)
        p = p_next
        g = g_next
        
    return seq
