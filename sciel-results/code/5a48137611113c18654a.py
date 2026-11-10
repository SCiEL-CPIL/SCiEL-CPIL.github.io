import time
import math

def construct():
    N = 123457
    
    # Precompute smallest prime factor (SPF) for all numbers up to N
    # This gives us O(log n) largest prime factor lookup
    spf = list(range(N + 1))
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    # Helper to get largest prime factor using SPF
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        max_p = 1
        while n > 1:
            p = spf[n]
            max_p = p
            while n % p == 0:
                n //= p
        return max_p
    
    # Systematic search over all splits a + b = N
    # We want to minimize max(P(a), P(b))
    # Strategy: for each possible max prime factor bound M,
    # check if there exists a split where both a and b have all prime factors <= M
    
    # First, let's do a full scan to find the optimal answer
    best_val = float('inf')
    best_pair = [1, N - 1]
    
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s kill
    
    # We can be smart: iterate a from 1 to N//2 (symmetry)
    # For each a, compute P(a) and P(b) where b = N - a
    # Track the minimum of max(P(a), P(b))
    
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
        b = N - a
        pa = largest_prime_factor(a)
        pb = largest_prime_factor(b)
        val = max(pa, pb)
        if val < best_val:
            best_val = val
            best_pair = [a, b]
    
    # Additional optimization: try to find even better by checking
    # if we can do better with a more targeted search
    # The above O(N) scan should find the true optimum since we check all splits
    
    return best_pair
