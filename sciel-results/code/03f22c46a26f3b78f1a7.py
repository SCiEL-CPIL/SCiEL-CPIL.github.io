import time
import math
import itertools
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 145s deadline, 5s safety margin
    
    # 1. Generate a list of small primes
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    primes = sieve(1000)
    prime_set = set(primes)
    
    # 2. Define fast tau_perp calculation for square-free numbers
    def tau_perp_squarefree(prime_list):
        """
        Calculate tau_perp for a square-free number given its prime factors.
        Divisors are products of subsets of primes.
        """
        k = len(prime_list)
        if k == 0:
            return 0
            
        # Generate all divisors
        # Bitmask approach for speed when k is not too large
        divs = []
        for mask in range(1 << k):
            d = 1
            for i in range(k):
                if mask & (1 << i):
                    d *= prime_list[i]
            divs.append(d)
            
        divs.sort()
        
        # Count adjacent coprime pairs
        # For square-free numbers, gcd(d_i, d_{i+1}) == 1 iff they share no prime factors
        # We can precompute the bitmask of prime factors for each divisor
        # Or just use math.gcd which is fast enough for 2^15 ~ 32768 divisors
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # 3. Strategy: Start with the product of the first k primes that fits in 10^18
    # Find max k
    max_k = 0
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        prod *= p
        max_k += 1
        
    # Initial candidate: first max_k primes
    initial_primes = primes[:max_k]
    best_score = tau_perp_squarefree(initial_primes)
    best_primes = list(initial_primes)
    
    # 4. Local Search (Hill Climbing)
    # Try swapping primes in the set to improve the score
    # We can only swap in primes that keep the product <= 10^18
    
    # Precompute product of current set
    current_prod = 1
    for p in best_primes:
        current_prod *= p
        
    # Candidate primes to consider for swapping in
    # We look at primes slightly larger than the smallest in the set, or smaller than the largest
    # Actually, to maintain the product constraint, if we swap p_out with p_in:
    # new_prod = current_prod * p_in / p_out <= 10^18
    # So p_in <= 10^18 * p_out / current_prod
    
    # Let's iterate through potential improvements
    # We'll do multiple passes until no improvement or time runs out
    
    improved = True
    while improved and time.time() < deadline - 10.0:
        improved = False
        # Sort current primes to keep track
        best_primes.sort()
        
        for i in range(len(best_primes)):
            if time.time() > deadline - 10.0:
                break
                
            p_out = best_primes[i]
            # Max value for p_in
            max_p_in = int(10**18 * p_out / current_prod)
            
            # Candidate primes are those in primes list that are:
            # 1. Not already in best_primes
            # 2. <= max_p_in
            # 3. Not equal to p_out
            
            # To be efficient, we can just iterate through primes list
            # and filter
            # Since primes list is sorted, we can stop when p > max_p_in
            for p_in in primes:
                if p_in > max_p_in:
                    break
                if p_in == p_out:
                    continue
                if p_in in best_primes:
                    continue
                    
                # Try the swap
                new_primes = best_primes[:]
                new_primes[i] = p_in
                new_primes.sort()
                
                # Optional: quick check if this swap is likely to be bad?
                # No, just evaluate.
                s = tau_perp_squarefree(new_primes)
                if s > best_score:
                    best_score = s
                    best_primes = new_primes
                    current_prod = current_prod * p_in // p_out
                    improved = True
                    break # Restart the outer loop with the new set
    
    # 5. Final Random Perturbation Search
    # If we have time, do some random restarts or perturbations
    if time.time() < deadline - 5.0:
        # Generate a few random valid sets
        # Strategy: Start with first k primes, then randomly swap
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            # Start from best_primes
            curr_primes = best_primes[:]
            curr_prod = current_prod
            
            # Do a few random swaps
            for _ in range(5):
                i = random.randint(0, len(curr_primes) - 1)
                p_out = curr_primes[i]
                max_p_in = int(10**18 * p_out / curr_prod)
                
                # Find candidates
                candidates = []
                for p in primes:
                    if p > max_p_in:
                        break
                    if p in curr_primes:
                        continue
                    candidates.append(p)
                
                if not candidates:
                    continue
                    
                p_in = random.choice(candidates)
                curr_primes[i] = p_in
                curr_prod = curr_prod * p_in // p_out
                curr_primes.sort()
                
            s = tau_perp_squarefree(curr_primes)
            if s > best_score:
                best_score = s
                best_primes = curr_primes
                current_prod = curr_prod

    # Construct final number
    result = 1
    for p in best_primes:
        result *= p
        
    return result
# EVOLVE-BLOCK-END
