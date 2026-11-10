import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i, v in enumerate(is_prime) if v]

    # Generate primes up to a larger limit to allow more search space
    primes = sieve(5000)
    
    def tau_perp(n, prime_factors):
        # prime_factors is a list of distinct primes
        divs = [1]
        for p in prime_factors:
            new_divs = []
            for d in divs:
                new_divs.append(d * p)
            divs.extend(new_divs)
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Initialize with smallest 15 primes
    initial_primes = []
    prod = 1
    for p in primes:
        if prod * p > 10**18:
            break
        initial_primes.append(p)
        prod *= p
        
    best_n = prod
    best_score = tau_perp(prod, initial_primes)
    best_primes = list(initial_primes)

    # Local Search
    # We try to swap any prime in the set with any other prime (larger or smaller)
    # that is not in the set, such that the new product <= 10^18.
    
    current_primes = list(initial_primes)
    current_prod = prod
    current_score = best_score
    
    improved = True
    iteration = 0
    while improved and time.time() < deadline - 10.0:
        improved = False
        iteration += 1
        # Shuffle indices to avoid bias
        indices = list(range(len(current_primes)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() > deadline - 5.0:
                break
            p_old = current_primes[idx]
            
            # Find candidate new primes
            # We can try primes from the global list.
            # To be efficient, we can bound the range.
            max_p_new = int((10**18 * p_old) / current_prod)
            
            # Search in primes list for p_new <= max_p_new and p_new not in current_primes (unless p_new == p_old)
            # We can optimize by pre-filtering, but a simple loop is ok for small k.
            best_swap_score = -1
            best_swap_prime = -1
            best_swap_prod = -1
            
            # Only consider primes that are different from current_primes
            current_set = set(current_primes)
            
            for p_new in primes:
                if p_new > max_p_new:
                    break
                if p_new == p_old:
                    continue
                if p_new in current_set:
                    continue
                
                # Evaluate
                new_prod = (current_prod // p_old) * p_new
                new_primes = list(current_primes)
                new_primes[idx] = p_new
                s_new = tau_perp(new_prod, new_primes)
                
                if s_new > best_swap_score:
                    best_swap_score = s_new
                    best_swap_prime = p_new
                    best_swap_prod = new_prod
            
            if best_swap_prime != -1 and best_swap_score > current_score:
                current_primes[idx] = best_swap_prime
                current_prod = best_swap_prod
                current_score = best_swap_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_n = current_prod
                    best_primes = list(current_primes)
                    
            if time.time() > deadline - 5.0:
                break

    return best_n
