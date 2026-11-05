import time
from math import isqrt

def construct():
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # safety margin before 90s kill

    # Generate all primes up to TARGET using sieve
    is_prime = [True] * (TARGET + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, isqrt(TARGET) + 1):
        if is_prime[i]:
            for j in range(i*i, TARGET + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, TARGET + 1) if is_prime[i]]
    
    def generate_smooth_numbers(limit, max_prime_idx):
        """Generate all numbers <= limit whose prime factors are all <= primes[max_prime_idx]"""
        smooth = set()
        # Use DFS/backtracking to generate smooth numbers
        def dfs(idx, current):
            if current > limit:
                return
            smooth.add(current)
            for i in range(idx, max_prime_idx + 1):
                if current * primes[i] > limit:
                    break
                dfs(i, current * primes[i])
        
        dfs(0, 1)
        return smooth
    
    # Incremental approach: try increasing B (max allowed prime factor)
    # For each B, generate all B-smooth numbers and check if any two sum to TARGET
    # The first B that works gives us the optimal answer
    
    best_a = 1
    best_b = TARGET - 1
    best_max_p = TARGET
    
    # Try each prime as potential max prime factor
    for pi, p in enumerate(primes):
        if time.time() - start_time > DEADLINE:
            break
        
        # Generate all p-smooth numbers up to TARGET
        smooth_a = generate_smooth_numbers(TARGET - 1, pi)
        
        # Check if any two smooth numbers sum to TARGET
        found = False
        for a in smooth_a:
            b = TARGET - a
            if b >= 1 and b in smooth_a:
                if a <= b:  # avoid duplicates
                    current_max = p
                    if current_max < best_max_p:
                        best_max_p = current_max
                        best_a = a
                        best_b = b
                        found = True
                        break
        
        if found:
            # Since we're going in increasing order of primes, this is optimal
            break
    
    return [best_a, best_b]
