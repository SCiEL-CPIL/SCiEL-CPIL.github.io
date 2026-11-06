import time
import math

def construct():
    deadline = time.time() + 55.0
    TARGET = 123457
    
    # List of primes to check K against
    # We don't need primes larger than TARGET, but practically K will be small
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307]
    
    # We will iterate through these primes as the bound K
    # For each K, we generate all K-smooth numbers <= TARGET
    
    # Optimization: Precompute primes up to some limit? 
    # The generate function needs the list of primes <= K.
    
    def generate_smooth_numbers(primes_list, limit):
        """
        Generates a set of all numbers <= limit whose prime factors are all in primes_list.
        """
        smooth = {1}
        # Use a queue or recursion? 
        # Simple approach: iterative expansion.
        # Start with [1]. 
        # For each prime p in primes_list:
        #   new_smooth = set()
        #   for n in smooth:
        #       val = n
        #       while val <= limit:
        #           new_smooth.add(val)
        #           val *= p
        #   smooth = smooth.union(new_smooth)
        # This works but might be slow if not careful. 
        # However, limit is 123457.
        
        # Better: recursive generation
        def gen(idx, current_val):
            if current_val > limit:
                return
            if idx == len(primes_list):
                smooth.add(current_val)
                return
            p = primes_list[idx]
            # Try multiplying by p^k
            val = current_val
            while val <= limit:
                gen(idx + 1, val)
                if val > limit // p:
                    break
                val *= p
        
        gen(0, 1)
        return smooth

    # We check K starting from 2
    # If we find a solution, return it.
    # If we exhaust primes or hit deadline, return best found or seed.
    
    best_a, best_b = 1, TARGET - 1
    best_score = 293 # From seed
    
    for i, K in enumerate(primes):
        if time.time() > deadline:
            break
            
        # Get primes <= K
        # Since primes list is sorted, primes[:i+1] are <= K
        current_primes = primes[:i+1]
        
        smooth_set = generate_smooth_numbers(current_primes, TARGET)
        
        # Check for pair
        found = False
        # To be deterministic and potentially faster, we can iterate
        # We just need any pair.
        for a in smooth_set:
            b = TARGET - a
            if b in smooth_set:
                # Found a valid pair with max prime factor <= K
                # Since we iterate K in increasing order, this is the minimal K
                return [a, b]
        
        # If no pair found, continue to next K
        
    # If loop finishes without finding a better solution (unlikely, as 293 is small)
    # Or if deadline hit, return the best we have (which is the seed if no improvement found yet)
    # Actually, if we didn't return inside the loop, it means no solution was found for K < 293?
    # Wait, the seed has score 293. 
    # If we check K=2,3,...,283 and find nothing, we should check K=293.
    # My primes list includes 293.
    # If we find a solution at K=293, it might be the same as seed or different.
    # But if we don't find anything up to the deadline, we return the seed.
    
    return [1, 123456]
