import math
import time
import random
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 148.0
    LIMIT = 10**18
    
    # Generate primes up to 200
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes_pool = sieve(200)
    
    # We will search over a subset of primes, say the first 10.
    # n = p1^e1 * p2^e2 * ...
    
    def eval_n(exponents, primes_subset):
        """
        Calculate tau_perp for n = prod(p_i^e_i).
        """
        # Generate divisors
        divs = [1]
        for i, p in enumerate(primes_subset):
            e = exponents[i]
            if e == 0:
                continue
            powers = [p**k for k in range(e + 1)]
            new_divs = []
            for d in divs:
                for pow_p in powers:
                    new_divs.append(d * pow_p)
            divs = new_divs
        
        if not divs:
            return 0
            
        divs.sort()
        
        score = 0
        prev = divs[0]
        for i in range(1, len(divs)):
            curr = divs[i]
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    # Initial state: Try a mix of exponents
    # Start with square-free first 10 primes
    k_primes = 10
    initial_primes = primes_pool[:k_primes]
    initial_exponents = [1] * k_primes
    
    # Check product
    prod = 1
    for i, p in enumerate(initial_primes):
        prod *= (p ** initial_exponents[i])
        
    if prod > LIMIT:
        # Reduce exponents
        while prod > LIMIT:
            # Reduce the largest prime's exponent
            for i in range(k_primes - 1, -1, -1):
                if initial_exponents[i] > 1:
                    initial_exponents[i] -= 1
                    prod //= initial_primes[i]
                    break
            else:
                # If all are 1, we can't reduce further without changing primes
                # But first 10 primes product is ~ 6.4e12, well within limit.
                break

    best_score = eval_n(initial_exponents, initial_primes)
    best_exponents = initial_exponents[:]
    best_primes = initial_primes[:]
    
    current_exponents = initial_exponents[:]
    current_primes = initial_primes[:]
    current_score = best_score
    
    # Local Search
    # Moves:
    # 1. Change exponent of one prime.
    # 2. Swap a prime with another from the pool.
    
    T = 10.0
    cooling = 0.99
    
    while time.time() < DEADLINE:
        # Pick a random move
        move_type = random.random()
        
        if move_type < 0.7:
            # Change exponent
            idx = random.randint(0, k_primes - 1)
            old_e = current_exponents[idx]
            new_e = old_e + random.choice([-1, 0, 1, 2, -2])
            if new_e < 1:
                new_e = 1
            # Check product limit
            # prod_new = prod_old / p^old_e * p^new_e
            # We need to track current product
            # For speed, we can just calculate new product if it's close
            # But we don't store prod in the loop efficiently.
            # Let's re-calculate product for the candidate.
            
            candidate_exponents = current_exponents[:]
            candidate_exponents[idx] = new_e
            
            prod = 1
            for i, p in enumerate(current_primes):
                prod *= (p ** candidate_exponents[i])
                if prod > LIMIT:
                    break
            
            if prod > LIMIT:
                continue
            
            score_new = eval_n(candidate_exponents, current_primes)
            delta = score_new - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_exponents = candidate_exponents
                current_score = score_new
                if current_score > best_score:
                    best_score = current_score
                    best_exponents = current_exponents[:]
        else:
            # Swap prime
            idx = random.randint(0, k_primes - 1)
            p_old = current_primes[idx]
            p_new = random.choice(primes_pool)
            if p_new in current_primes:
                continue
            if p_new == p_old:
                continue
            
            candidate_primes = current_primes[:]
            candidate_primes[idx] = p_new
            
            prod = 1
            for i, p in enumerate(candidate_primes):
                prod *= (p ** current_exponents[i])
                if prod > LIMIT:
                    break
            
            if prod > LIMIT:
                continue
            
            score_new = eval_n(current_exponents, candidate_primes)
            delta = score_new - current_score
            
            if delta >= 0 or random.random() < math.exp(delta / T):
                current_primes = candidate_primes
                current_score = score_new
                if current_score > best_score:
                    best_score = current_score
                    best_primes = candidate_primes[:]
                    
        T *= cooling
        if T < 0.1:
            T = 0.1

    # Construct n from best
    n = 1
    for i, p in enumerate(best_primes):
        n *= (p ** best_exponents[i])
        
    return n

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
