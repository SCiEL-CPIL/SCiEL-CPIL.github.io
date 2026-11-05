"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin for 150s limit
    
    # 1. Generate enough primes
    # Product of first 15 primes is ~6e17, so 15 primes is a good start.
    # We might need up to ~20-25 primes if we use small exponents, but 
    # since n <= 10^18, the number of distinct prime factors is limited.
    # Max distinct primes: product of first k primes <= 10^18.
    # 2*3*5*...
    # Let's generate first 50 primes just in case, though we likely won't use many.
    
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime[i]:
                for j in range(i*i, n + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(200)
    
    # 2. Initial candidate: Product of first 15 primes (squarefree)
    # 2*3*5*7*11*13*17*19*23*29*31*37*41*43*47 = 614889782588491410
    # This is < 10^18.
    # Score for squarefree n with k primes:
    # Divisors are 2^k. 
    # Adjacent pairs in sorted order of squarefree divisors.
    # For squarefree n, gcd(d_i, d_{i+1}) = 1 iff d_i and d_{i+1} share no prime factors.
    # In the sorted list of divisors of a squarefree number, consecutive divisors often share factors.
    # Example n=6 (2,3): divs 1,2,3,6. Pairs: (1,2)g1, (2,3)g1, (3,6)g3. Score 2.
    # Example n=30 (2,3,5): divs 1,2,3,5,6,10,15,30.
    # (1,2)1, (2,3)1, (3,5)1, (5,6)1, (6,10)2, (10,15)5, (15,30)15. Score 4.
    # Generally, squarefree numbers have decent scores but maybe not maximal.
    
    init_exponents = [1] * 15
    # Verify product
    n = 1
    for i in range(15):
        n *= primes[i]
    if n > 10**18:
        # Fallback to fewer primes if logic changes
        while n > 10**18 and len(init_exponents) > 0:
            init_exponents.pop()
            n //= primes[len(init_exponents)]
        
    best_exponents = init_exponents[:]
    best_score = 0
    
    # Helper to calculate score for given exponents
    def calc_score(exps):
        # Reconstruct n
        n = 1
        for i, e in enumerate(exps):
            if e > 0:
                n *= primes[i] ** e
        if n > 10**18 or n < 2:
            return -1, n
        
        # Get divisors
        # Efficient divisor generation
        divs = [1]
        for i, e in enumerate(exps):
            if e == 0:
                continue
            p = primes[i]
            new_divs = []
            p_pow = 1
            for _ in range(e + 1):
                for d in divs:
                    new_divs.append(d * p_pow)
                p_pow *= p
            divs = new_divs
        
        divs.sort()
        
        # Calculate score
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score, n

    # Initial evaluation
    best_score, _ = calc_score(best_exponents)
    
    # Local Search
    # Moves: Increase exponent of prime i by 1, or Decrease by 1 (if > 0)
    # We iterate until time runs out or no improvement is found in a full pass.
    
    # Optimization: Precompute prime powers? 
    # For local search, we just need to evaluate neighbors.
    # Number of neighbors is ~2 * num_primes.
    # Cost of calc_score is O(tau(n)). For n~10^18, tau(n) can be up to ~100k.
    # 100k gcds is fast in Python? ~0.01s?
    # If we do 1000 steps, that's 10s.
    # We need to be careful.
    
    # Let's optimize calc_score slightly by avoiding full sort if possible? 
    # No, divisors must be sorted to find adjacent pairs.
    # However, we can stop early if n is too large.
    
    # To speed up, we can limit the number of primes we consider.
    # Primes beyond the 20th are unlikely to be in the optimal solution for n<=10^18 
    # because their contribution to divisor count is small and they eat into the product budget.
    # Actually, small primes allow for higher exponents. Large primes force exponents to be 0 or 1.
    # Let's restrict search to first 20 primes.
    max_primes = 20
    
    # Ensure best_exponents has length max_primes
    while len(best_exponents) < max_primes:
        best_exponents.append(0)
    best_exponents = best_exponents[:max_primes]
    
    # Re-evaluate with restricted list
    best_score, _ = calc_score(best_exponents)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(max_primes):
            # Try increasing
            if best_exponents[i] + 1 <= 20: # Cap exponent to avoid huge n
                new_exps = best_exponents[:]
                new_exps[i] += 1
                s, n_val = calc_score(new_exps)
                if s > best_score:
                    best_score = s
                    best_exponents = new_exps
                    improved = True
                    if time.time() > deadline:
                        break
            # Try decreasing
            if best_exponents[i] > 0:
                new_exps = best_exponents[:]
                new_exps[i] -= 1
                s, n_val = calc_score(new_exps)
                if s > best_score:
                    best_score = s
                    best_exponents = new_exps
                    improved = True
                    if time.time() > deadline:
                        break
        if time.time() > deadline:
            break
            
    # Final n
    final_n = 1
    for i in range(max_primes):
        if best_exponents[i] > 0:
            final_n *= primes[i] ** best_exponents[i]
            
    if final_n > 10**18 or final_n < 2:
        # Fallback to a known valid number
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47
        
    return final_n
# EVOLVE-BLOCK-END
