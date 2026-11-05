import sys
from functools import lru_cache
import time

def construct():
    start_time = time.time()
    LIMIT = 10000
    DEADLINE = start_time + 85.0  # Safety margin before 90s kill
    
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
    best_seq = []
    
    # 4. Iterate over first two primes
    n = len(primes)
    found_improvement = False
    
    # We will iterate through pairs. To find the absolute best, we need to check all.
    # However, we can prune.
    # If 1 + calc_len(p2, g1) <= best_len, we don't update.
    
    for i in range(n):
        p1 = primes[i]
        # If even the max possible length starting from p1 is less than best_len, we might prune?
        # Max possible length is bounded by (LIMIT - p1) / min_gap. Min gap is 2 (for 2->3).
        # But gaps increase.
        
        for j in range(i + 1, n):
            p2 = primes[j]
            g1 = p2 - p1
            
            # Quick check: if potential max is low?
            # calc_len is cached, so repeated calls are fast.
            l = 1 + calc_len(p2, g1)
            
            if l > best_len:
                best_len = l
                best_seq = [p1]
                p = p2
                g = g1
                while True:
                    best_seq.append(p)
                    target = p + g
                    if target > LIMIT:
                        break
                    p_next = next_prime_ge[target]
                    if p_next == -1 or p_next > LIMIT:
                        break
                    g_next = p_next - p
                    p = p_next
                    g = g_next
                found_improvement = True
                
            # Time check
            if time.time() > DEADLINE:
                # If we have a valid sequence, return it.
                if best_seq:
                    return best_seq
                # Fallback
                return [primes[0]] if primes else []
    
    if best_seq:
        return best_seq
        
    # Fallback: return first prime or empty
    return [primes[0]] if primes else []
