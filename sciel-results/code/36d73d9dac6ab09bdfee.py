import math
import time
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Helper to calculate tau_perp(n) efficiently
    def calc_score(n):
        # Factorize n
        factors = {}
        temp_n = n
        p = 2
        while p * p <= temp_n:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors[p] = count
            p += 1 if p == 2 else 2
        if temp_n > 1:
            factors[temp_n] = 1
        
        # Generate divisors
        divs = [1]
        for p, e in factors.items():
            new_divs = []
            p_powers = [1]
            for _ in range(e):
                p_powers.append(p_powers[-1] * p)
            for d in divs:
                for pp in p_powers:
                    new_divs.append(d * pp)
            divs = new_divs
        
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score, factors

    # Initial candidate: Product of first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    current_n = 1
    for p in primes:
        current_n *= p
        
    current_score, current_factors = calc_score(current_n)
    
    # Local Search: Greedy Hill Climbing
    # We explore neighbors by changing exponents of existing primes or adding/removing primes.
    # To keep it fast, we limit the search depth and the range of changes.
    
    best_n = current_n
    best_score = current_score
    best_factors = current_factors
    
    # List of primes to consider for addition (up to ~100)
    candidate_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will perform a few passes of local search.
    # In each pass, we try to improve the score by a single move.
    # Moves:
    # 1. Increase exponent of a prime p in current_factors by 1
    # 2. Decrease exponent of a prime p in current_factors by 1 (if > 0)
    # 3. Add a new prime q not in current_factors
    # 4. Remove a prime p from current_factors (set exponent to 0)
    
    improved = True
    max_iterations = 100
    iteration = 0
    
    while improved and time.time() < deadline and iteration < max_iterations:
        improved = False
        iteration += 1
        
        # Get current primes and exponents
        cur_primes = list(best_factors.keys())
        
        # Try all possible single-step moves
        best_move_score = best_score
        best_move_n = best_n
        best_move_factors = best_factors
        
        # 1. Increase exponent
        for p in cur_primes:
            new_n = best_n * p
            if new_n <= 10**18:
                s, f = calc_score(new_n)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = f
                    improved = True
        
        # 2. Decrease exponent
        for p in cur_primes:
            if best_factors[p] > 1:
                new_n = best_n // p
                s, f = calc_score(new_n)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = f
                    improved = True
        
        # 3. Add new prime
        # We should try adding smaller primes first as they are more likely to fit and change structure significantly
        for q in candidate_primes:
            if q not in best_factors:
                new_n = best_n * q
                if new_n <= 10**18:
                    s, f = calc_score(new_n)
                    if s > best_move_score:
                        best_move_score = s
                        best_move_n = new_n
                        best_move_factors = f
                        improved = True
        
        # 4. Remove prime
        for p in cur_primes:
            if best_factors[p] == 1:
                new_n = best_n // p
                s, f = calc_score(new_n)
                if s > best_move_score:
                    best_move_score = s
                    best_move_n = new_n
                    best_move_factors = f
                    improved = True
        
        if improved:
            best_n = best_move_n
            best_score = best_move_score
            best_factors = best_move_factors
    
    return best_n

# To satisfy the requirement of keeping function names/signatures, 
# although the task says "Keep the function names and signatures of the seed program", 
# the seed program only has construct().
