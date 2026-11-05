import time
import sys

def construct():
    start_time = time.time()
    deadline = 85.0  # 5s margin
    
    # 1. Sieve of Eratosthenes to get primes up to 10^6
    limit = 1000000
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    primes = [i for i, flag in enumerate(is_prime) if flag]
    
    # 2. Iterate over prime bounds
    # We want the smallest B such that there exist n, n+1 in [10000, 1000000] 
    # which are B-smooth.
    
    # To generate B-smooth numbers in [10000, 1000000], we can use a recursive approach
    # or a heap. Given the range and small number of primes for small B, recursion is fine.
    
    def generate_smooth_numbers(primes_subset, low, high, max_val):
        """
        Generates all numbers <= max_val that are composed only of primes in primes_subset.
        Filters for [low, high].
        """
        results = []
        def backtrack(idx, current_val):
            if current_val > max_val:
                return
            if current_val >= low:
                results.append(current_val)
            
            if idx == len(primes_subset):
                return
            
            p = primes_subset[idx]
            # Try multiplying by p^k
            val = current_val
            while val <= max_val:
                backtrack(idx + 1, val)
                if val > max_val // p:
                    break
                val *= p
        
        backtrack(0, 1)
        return results

    # We iterate k from 1 to len(primes)
    # But we can stop early if time is running out.
    
    best_n = None
    best_score = float('inf')
    
    # Optimization: 
    # If we find a pair for bound B, that B is the answer because we check B in increasing order.
    # The "score" is the largest prime factor, which is exactly B (the largest prime used).
    # Wait, is it possible that n or n+1 has a prime factor > B? No, by definition of B-smooth.
    # So the largest prime factor of n(n+1) is at most B.
    # If we find a pair that is B-smooth, the largest prime factor is <= B.
    # Since we iterate B in increasing order of primes, the first B that yields a pair 
    # gives the minimal possible largest prime factor.
    
    for k in range(1, len(primes)):
        if time.time() > deadline:
            break
            
        current_bound = primes[k-1]
        primes_subset = primes[:k]
        
        # Generate smooth numbers in [10000, 1000000]
        smooth_nums = generate_smooth_numbers(primes_subset, 10000, 1000000, 1000000)
        
        if not smooth_nums:
            continue
            
        smooth_set = set(smooth_nums)
        
        # Check for consecutive pairs
        found_pair = False
        for n in smooth_set:
            if n + 1 in smooth_set:
                # Found a pair
                # The largest prime factor of n(n+1) is at most current_bound.
                # Actually, it could be less if the numbers don't use the largest prime.
                # But we are minimizing the *bound* B such that they are B-smooth.
                # The question asks to minimize the *actual* largest prime factor.
                # Let L(n) be the largest prime factor of n(n+1).
                # We want min L(n).
                # If n, n+1 are B-smooth, then L(n) <= B.
                # By iterating B from small to large, the first B where a pair exists 
                # ensures that no pair exists with largest prime factor < B.
                # Therefore, the minimal possible value for L(n) is achieved by this pair?
                # Not necessarily. The pair might have L(n) < B.
                # Example: B=5. Pair 8, 9. L(8*9) = L(72) = 3.
                # If we found a pair with B=3 earlier, we would have stopped.
                # Since we didn't stop at B=2, no pair is 2-smooth.
                # So min L(n) is at least 3.
                # If we find a pair at B=3, L(n) is 3 (since 2-smooth failed).
                # So the first B that yields a pair IS the minimal largest prime factor.
                
                return n
        
    # If no pair found within time or primes, return a default
    # This should not happen for this range, but safe to have
    return 10000
