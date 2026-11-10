import math
import time

def generate_smooth_numbers(n, primes):
    """Generate all numbers <= n that have no prime factors other than those in primes."""
    smooths = set()
    # Recursive generation
    def gen(current, idx):
        if current > n:
            return
        if idx == len(primes):
            smooths.add(current)
            return
        p = primes[idx]
        val = current
        while val <= n:
            gen(val, idx + 1)
            val *= p
        # Also include the case where we don't use this prime
        # This is handled by the loop structure if we are careful, 
        # but standard recursive generation usually multiplies.
        # Let's use a different approach: iterative multiplication.
        
    # Better approach:
    # Start with [1]
    # For each prime, multiply existing elements by powers of the prime.
    # But this is tricky to do efficiently without duplicates.
    
    # Simple DFS:
    def dfs(current, idx):
        if current > n:
            return
        if idx == len(primes):
            smooths.add(current)
            return
        p = primes[idx]
        # Option 1: don't use p
        dfs(current, idx + 1)
        # Option 2: use p
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
    best_val = 643 # Value from seed
    
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
                # We found a pair where both are smooth wrt 'primes'
                # The max prime factor is at most primes[-1]
                # We should verify the actual max prime factor to be precise
                pa = largest_prime_factor(a)
                pb = largest_prime_factor(b)
                val = max(pa, pb)
                
                if val < best_val:
                    best_val = val
                    best_pair = [a, b]
                    found = True
                    # Since we are iterating primes in increasing order, 
                    # the first found set might not give the absolute min if 
                    # a smaller set didn't work, but a larger set does.
                    # However, if [2,3] works, max factor is 3.
                    # If [2,3] fails, [2,3,5] works, max factor is 5.
                    # We want the minimum max factor.
                    # So we should stop at the first prime set that yields a solution?
                    # Yes, because any solution in [2,3] has P<=3.
                    # Any solution in [2,3,5] has P<=5.
                    # If [2,3] fails, no solution with P<=3 exists.
                    # So the first successful set gives the minimal max prime factor.
                    break
        
        if found:
            # We can break out of the outer loop too, as we found the minimal K
            # for this specific N.
            # Wait, is it possible that a solution with P=5 exists but P=3 doesn't?
            # Yes.
            # Is it possible that a solution with P=7 exists but P=5 doesn't?
            # Yes.
            # So we must iterate from smallest prime set to largest.
            # The first hit is the optimal K.
            break

    # If no smooth solution found in the sets, fall back to brute force or return best found
    if best_val == 643:
        # Fallback to original logic if needed, but 11-smooth should exist
        # The current program found 11, so [2,3,5,7,11] should work.
        pass

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
