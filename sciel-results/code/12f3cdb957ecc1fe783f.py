import math
import time

def generate_smooth_numbers(n, primes):
    """Generate all numbers <= n that have no prime factors other than those in primes."""
    smooths = set()
    
    def dfs(current, idx):
        if current > n:
            return
        if idx == len(primes):
            smooths.add(current)
            return
        p = primes[idx]
        # Option 1: don't use p
        dfs(current, idx + 1)
        # Option 2: use p multiple times
        val = current
        while val * p <= n:
            val *= p
            dfs(val, idx + 1)
            
    dfs(1, 0)
    return smooths

def construct():
    N = 123457
    
    # Try increasing sets of primes
    prime_sets = [
        [2, 3],
        [2, 3, 5],
        [2, 3, 5, 7],
        [2, 3, 5, 7, 11],
        [2, 3, 5, 7, 11, 13]
    ]
    
    best_pair = [1, N - 1]
    best_val = largest_prime_factor(N - 1)
    
    start_time = time.time()
    deadline = start_time + 50
    
    for primes in prime_sets:
        if time.time() > deadline:
            break
            
        smooths = generate_smooth_numbers(N, primes)
        
        # Check for pair
        found = False
        for a in smooths:
            b = N - a
            if b in smooths:
                pa = largest_prime_factor(a)
                pb = largest_prime_factor(b)
                val = max(pa, pb)
                
                if val < best_val:
                    best_val = val
                    best_pair = [a, b]
                    found = True
                    break
        
        if found:
            break

    return best_pair

def largest_prime_factor(n):
    if n == 1:
        return 1
    max_factor = 1
    while n % 2 == 0:
        max_factor = 2
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            max_factor = f
            n //= f
        f += 2
    if n > 1:
        max_factor = n
    return max_factor
