"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute primes up to sqrt(1000000) for trial division
    2. Use segmented approach: for each candidate smoothness bound B,
       find numbers in [10000, 1000000] that are B-smooth
    3. Check consecutive pairs where both n and n+1 are B-smooth
    4. For the best candidates, compute exact largest prime factor via trial division
    5. Track the best (n, largest_prime_factor) found
    """
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline with 5s safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Precompute primes up to sqrt(N_MAX) for trial division
    limit = int(math.isqrt(N_MAX)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    def largest_prime_factor(n):
        """Compute the largest prime factor of n using trial division."""
        if n <= 1:
            return 1
        result = 1
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                result = p
                while n % p == 0:
                    n //= p
        if n > 1:
            result = max(result, n)
        return result
    
    # Try different smoothness bounds
    # A number is B-smooth if all its prime factors are <= B
    # We want to find n where both n and n+1 are B-smooth for small B
    
    best_n = N_MIN
    best_lpf = largest_prime_factor(N_MIN * (N_MIN + 1))
    
    # For efficiency, we'll check smoothness bounds in increasing order
    # and stop when we can't improve further
    # 
    # Key insight: if we find n where both n and n+1 are B-smooth,
    # then largest prime factor of n(n+1) <= B
    # So we want the smallest B such that there exist consecutive B-smooth numbers
    
    # Generate all primes up to a reasonable bound for smoothness checking
    # We'll check B values: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, ...
    
    # Precompute which numbers up to N_MAX are B-smooth for various B
    # Use a segmented sieve approach
    
    def find_smooth_consecutive(B, primes_for_smooth):
        """Find all n in [N_MIN, N_MAX-1] where both n and n+1 are B-smooth."""
        # A number is B-smooth if it has no prime factor > B
        # Equivalent: after dividing out all prime factors <= B, the remainder is 1
        
        # For efficiency, we can use the fact that if n is B-smooth,
        # then n = product of primes <= B raised to some powers
        
        # Generate all B-smooth numbers up to N_MAX
        # This is feasible for small B
        
        smooth_numbers = set()
        
        def gen_smooth(primes_list, max_val, current=1, idx=0):
            if current > max_val:
                return
            if idx == len(primes_list):
                if current >= N_MIN and current <= N_MAX:
                    smooth_numbers.add(current)
                return
            p = primes_list[idx]
            # Multiply by p^k for k >= 0
            val = current
            while val <= max_val:
                gen_smooth(primes_list, max_val, val, idx + 1)
                if val > max_val // p:
                    break
                val *= p
        
        gen_smooth(primes_for_smooth, N_MAX)
        
        # Now check for consecutive pairs
        candidates = []
        for n in range(N_MIN, N_MAX):
            if n in smooth_numbers and (n + 1) in smooth_numbers:
                candidates.append(n)
        
        return candidates
    
    # Try increasing B values
    # For each B, get the primes <= B and find consecutive B-smooth numbers
    # The first B where we find a candidate gives us the answer
    
    # But we need to be careful: the largest prime factor of n(n+1) might be
    # smaller than B even if n and n+1 are B-smooth (if they don't actually use the largest prime)
    # So we should compute the actual largest prime factor for candidates
    
    # Let's try B values in order
    # Primes up to 100 should be sufficient
    primes_up_to_100 = [p for p in primes if p <= 100]
    
    # For each subset of primes (in increasing order of max prime), find consecutive smooth numbers
    # Start with small sets of primes
    
    # Actually, let's be more systematic:
    # For B in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    #   Find all n in [N_MIN, N_MAX-1] where both n and n+1 are B-smooth
    #   For each such n, compute largest_prime_factor(n*(n+1))
    #   Track the minimum
    
    # To make this efficient, we'll generate B-smooth numbers incrementally
    
    # For small B, the number of B-smooth numbers up to 1000000 is manageable
    # For B=2: powers of 2 only
    # For B=3: 2^a * 3^b
    # etc.
    
    # Let's use a different approach: for each B, generate all B-smooth numbers
    # and check for consecutive pairs
    
    # Precompute for each B value
    B_values = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]
    
    for B in B_values:
        if time.time() > deadline:
            break
        
        # Get primes <= B
        primes_for_B = [p for p in primes if p <= B]
        
        # Generate all B-smooth numbers in [N_MIN, N_MAX]
        smooth_set = set()
        
        def gen_smooth_recursive(primes_list, max_val, current=1, idx=0):
            if current > max_val:
                return
            if idx == len(primes_list):
                if N_MIN <= current <= N_MAX:
                    smooth_set.add(current)
                return
            p = primes_list[idx]
            val = current
            while val <= max_val:
                gen_smooth_recursive(primes_list, max_val, val, idx + 1)
                if val > max_val // p:
                    break
                val *= p
        
        gen_smooth_recursive(primes_for_B, N_MAX)
        
        # Check for consecutive pairs
        found_any = False
        for n in range(N_MIN, N_MAX):
            if n in smooth_set and (n + 1) in smooth_set:
                found_any = True
                lpf = largest_prime_factor(n * (n + 1))
                if lpf < best_lpf:
                    best_lpf = lpf
                    best_n = n
        
        # If we found a candidate with lpf == B (the current bound), 
        # we might be able to stop if no smaller B could work
        # But actually, we should continue to see if we can do better
        # However, since we're going in increasing B order, the first B where
        # we find a candidate likely gives the best answer
        # But the actual lpf might be less than B, so we should check all B values
        
        # Optimization: if best_lpf <= B, then no larger B can improve
        if best_lpf <= B:
            break
    
    # Final verification
    final_lpf = largest_prime_factor(best_n * (best_n + 1))
    if final_lpf < best_lpf:
        best_lpf = final_lpf
    
    return best_n
# EVOLVE-BLOCK-END
