import time
import math
import random

def construct():
    start_time = time.time()
    LIMIT = 10**18
    DEADLINE = start_time + 140.0  # Safety margin

    # List of first 30 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 
              109, 113]

    def calc_score(factors):
        """
        factors: list of (prime, exponent)
        Returns the score tau_perp(n)
        """
        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            for _ in range(e):
                current_divs = [d * p for d in current_divs]
                divs.extend(current_divs)
        
        divs.sort()
        
        score = 0
        prev = divs[0]
        for curr in divs[1:]:
            if math.gcd(prev, curr) == 1:
                score += 1
            prev = curr
        return score

    best_score = 0
    best_n = 2 # Fallback
    
    # Strategy 1: Systematic search over exponents of first few primes
    # We assume the optimal number has the form:
    # 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * (product of subsequent primes with exp 1)
    
    # We will iterate over a, b, c, d, e, f. 
    # g and h will be 1 initially, but we can try increasing them if there is space.
    # The "tail" primes will be added greedily.
    
    # To limit the search space, we note that increasing exponents of larger primes
    # is less effective than increasing exponents of smaller primes or adding new primes.
    # So we focus on a,b,c,d having higher potential, and e,f,g,h being small.
    
    max_a = 15
    max_b = 10
    max_c = 7
    max_d = 5
    max_e = 4
    max_f = 3
    
    # Precompute powers for speed
    pow2 = [2**i for i in range(max_a+1)]
    pow3 = [3**i for i in range(max_b+1)]
    pow5 = [5**i for i in range(max_c+1)]
    pow7 = [7**i for i in range(max_d+1)]
    pow11 = [11**i for i in range(max_e+1)]
    pow13 = [13**i for i in range(max_f+1)]

    # We will iterate through combinations of a,b,c,d,e,f
    # For each, we calculate the base product.
    # Then we multiply by 17, 19, 23... until we exceed LIMIT.
    # We also try variations where we stop adding primes earlier to allow higher exponents on 17, 19 etc?
    # Actually, the greedy addition of square-free primes is a strong heuristic.
    # Let's stick to: Fix exponents of first 6 primes. Greedily add next primes.
    
    evaluated = set()
    
    for a in range(1, max_a + 1):
        if time.time() > DEADLINE - 10:
            break
        for b in range(1, max_b + 1):
            base23 = pow2[a] * pow3[b]
            if base23 > LIMIT:
                break
            for c in range(1, max_c + 1):
                base235 = base23 * pow5[c]
                if base235 > LIMIT:
                    break
                for d in range(1, max_d + 1):
                    base2357 = base235 * pow7[d]
                    if base2357 > LIMIT:
                        break
                    for e in range(1, max_e + 1):
                        base235711 = base2357 * pow11[e]
                        if base235711 > LIMIT:
                            break
                        for f in range(1, max_f + 1):
                            base_core = base235711 * pow13[f]
                            if base_core > LIMIT:
                                break
                            
                            # Now, greedily add primes starting from index 6 (17)
                            # primes[6] = 17
                            current_n = base_core
                            factors = [
                                (2, a), (3, b), (5, c), (7, d), (11, e), (13, f)
                            ]
                            
                            # Try adding 17, 19, 23...
                            for i in range(6, len(primes)):
                                p = primes[i]
                                if current_n * p > LIMIT:
                                    break
                                current_n *= p
                                factors.append((p, 1))
                                
                                # Evaluate this candidate
                                # Optimization: Only evaluate if we haven't seen this n
                                if current_n not in evaluated:
                                    evaluated.add(current_n)
                                    score = calc_score(factors)
                                    if score > best_score:
                                        best_score = score
                                        best_n = current_n
                                        
                            # Also consider the case where we stop adding primes earlier 
                            # and increase exponent of 17?
                            # This is complex. The greedy square-free tail is usually good.
                            # Let's also try a few specific "high exponent" variants for 17 and 19
                            # if there is space.
                            
                            # Variant: Increase exponent of 17 if possible
                            if current_n * 17 <= LIMIT:
                                # This implies we didn't add 17 in the loop above? 
                                # No, the loop adds 17 first. 
                                # If we added 17, current_n has factor 17^1.
                                # Can we make it 17^2?
                                if 17 in [f[0] for f in factors]:
                                    # Find index of 17
                                    idx_17 = next(i for i, (p, _) in enumerate(factors) if p == 17)
                                    if factors[idx_17][1] == 1:
                                        # Try doubling 17
                                        new_n = current_n * 17
                                        if new_n <= LIMIT:
                                            new_factors = factors.copy()
                                            new_factors[idx_17] = (17, 2)
                                            if new_n not in evaluated:
                                                evaluated.add(new_n)
                                                score = calc_score(new_factors)
                                                if score > best_score:
                                                    best_score = score
                                                    best_n = new_n

    # Strategy 2: Local Search on the best found
    # Try to swap primes in the tail or adjust exponents of small primes
    if best_n > 2:
        # Factor best_n to get current exponents
        temp_n = best_n
        factors = []
        for p in primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                cnt = 0
                while temp_n % p == 0:
                    temp_n //= p
                    cnt += 1
                factors.append((p, cnt))
        if temp_n > 1:
            factors.append((temp_n, 1))
        
        current_factors = factors
        current_score = best_score
        
        # Try small perturbations
        # 1. Increase exponent of smallest prime if it fits
        # 2. Decrease exponent of largest prime in tail and increase exponent of a smaller prime?
        
        # Simple hill climbing: try to increase exponent of 2, 3, 5 if it fits
        for p in [2, 3, 5]:
            if p in [f[0] for f in current_factors]:
                idx = next(i for i, (pp, _) in enumerate(current_factors) if pp == p)
                old_exp = current_factors[idx][1]
                new_exp = old_exp + 1
                # Calculate new n
                # n_new = n_old * p
                if best_n * p <= LIMIT:
                    new_factors = current_factors.copy()
                    new_factors[idx] = (p, new_exp)
                    new_n = best_n * p
                    if new_n not in evaluated:
                        evaluated.add(new_n)
                        score = calc_score(new_factors)
                        if score > current_score:
                            current_score = score
                            current_factors = new_factors
                            best_n = new_n
                            best_score = current_score

    # Final safety check
    if best_n < 2 or best_n > LIMIT:
        best_n = 614889782588491410 # 2*3*...*47

    return best_n

