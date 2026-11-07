"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

import math

def construct():
    """
    Returns a list of distinct integers from 1..256 such that every pair is coprime,
    maximizing the sum.
    """
    N = 256
    
    # Step 1: Find all primes up to N
    is_prime = [True] * (N + 1)
    is_prime[0] = False
    if N >= 1:
        is_prime[1] = False
        
    for i in range(2, int(N**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, N + 1, i):
                is_prime[j] = False
                
    primes = [i for i in range(2, N + 1) if is_prime[i]]
    prime_set = set(primes)
    
    # Step 2: For each prime, find the largest power p^k <= N
    prime_to_power = {}
    for p in primes:
        power = p
        next_power = p * p
        while next_power <= N:
            power = next_power
            next_power *= p
        prime_to_power[p] = power
        
    # Initial set: 1 and all largest prime powers
    current_set = [1]
    current_set.extend(prime_to_power.values())
    
    # Track which primes are "used" by which number in current_set
    # Initially, each prime p is used by prime_to_power[p]
    # 1 uses no primes.
    used_primes = {} # prime -> number using it
    
    for p in primes:
        used_primes[p] = prime_to_power[p]
        
    # Helper to get prime factors
    def get_prime_factors(n):
        factors = []
        d = 2
        while d * d <= n:
            if n % d == 0:
                factors.append(d)
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            factors.append(n)
        return factors

    # Precompute factors for all composites
    composites = [n for n in range(N, 1, -1) if not is_prime[n] and n != 1]
    composite_factors = {n: get_prime_factors(n) for n in composites}
    
    # Iterative Local Search
    # We allow swaps:
    # 1. Replace a set of prime powers with one composite.
    # 2. Replace one existing composite with another composite (if beneficial).
    
    improved = True
    max_iterations = 100
    
    while improved and max_iterations > 0:
        improved = False
        max_iterations -= 1
        
        # Try to add/swap in each composite
        for n in composites:
            if n in current_set:
                continue
                
            factors = composite_factors[n]
            if not factors:
                continue
                
            # Identify numbers to remove:
            # 1. Prime powers for primes in `factors` that are currently owned by prime powers.
            # 2. Existing composites that use primes in `factors`.
            
            numbers_to_remove = []
            sum_removed = 0
            valid_swap = True
            
            for p in factors:
                if p in used_primes:
                    owner = used_primes[p]
                    # If owner is a prime power, it's in current_set.
                    # If owner is a composite, it's in current_set.
                    # We need to remove ALL numbers that share ANY prime with n.
                    # Since we iterate factors of n, we collect all owners.
                    if owner not in numbers_to_remove:
                        numbers_to_remove.append(owner)
                        sum_removed += owner
                else:
                    # Prime p is not used? 
                    # This shouldn't happen if we started with all prime powers and only swap.
                    # Unless we removed a number and didn't free the prime? 
                    # My logic above updates used_primes.
                    valid_swap = False
                    break
                    
            if not valid_swap:
                continue
                
            # Calculate gain
            gain = n - sum_removed
            if gain <= 0:
                continue
                
            # Check coprimality with the REST of the set
            # The rest is current_set - numbers_to_remove
            # We need n to be coprime to all x in (current_set - numbers_to_remove)
            
            # Optimization: 
            # n's factors are `factors`.
            # Any x in current_set that is NOT in numbers_to_remove must NOT share any factor with n.
            # If x shares a factor p with n, then p is in `factors`.
            # Then p is in used_primes, and used_primes[p] would be x (or someone else).
            # If used_primes[p] is x, then x would be in numbers_to_remove.
            # So if x is NOT in numbers_to_remove, it does NOT share any prime with n.
            # Thus, coprimality is GUARANTEED by the construction of numbers_to_remove!
            # We don't need to check gcd(n, x) explicitly if used_primes is accurate.
            
            # Perform swap
            for num in numbers_to_remove:
                # Update used_primes: free the primes owned by num
                # We need to know the factors of num to free them.
                # If num is a prime power, its factor is the prime itself.
                # If num is a composite, use composite_factors.
                if num == 1:
                    continue
                if num in composite_factors:
                    f_num = composite_factors[num]
                else:
                    # It's a prime power. Find its prime.
                    # We can just iterate primes? Or store mapping.
                    # Let's find p such that prime_to_power[p] == num
                    # Since num is a prime power, it's unique.
                    f_num = [p for p in primes if prime_to_power[p] == num]
                    if not f_num:
                        # Should not happen
                        continue
                        
                for p in f_num:
                    if p in used_primes and used_primes[p] == num:
                        del used_primes[p]
                        
                current_set.remove(num)
                
            # Add n
            current_set.append(n)
            for p in factors:
                used_primes[p] = n
                
            improved = True

    return current_set

# EVOLVE-BLOCK-END
