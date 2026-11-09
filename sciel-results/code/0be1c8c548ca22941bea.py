import math
import time

def construct():
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Primes to consider for the base product
    # We start with the product of the first 15 primes (up to 47)
    # and then optimize exponents of the first few primes.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    # Base product of primes from index 5 onwards (13, 17, ..., 47)
    # We will optimize exponents for 2, 3, 5, 7, 11
    base_primes = primes[5:]
    base_product = 1
    for p in base_primes:
        base_product *= p
        
    # We have budget 10^18
    # n = 2^a * 3^b * 5^c * 7^d * 11^e * base_product
    # Let's search for a, b, c, d, e
    
    best_n = base_product
    best_score = 0
    
    # Helper to calculate score (tau_perp)
    # For performance, we only calculate exact score for candidate n
    # that pass a quick divisor count heuristic or are near the end.
    # However, the score is not just divisor count.
    # But for these numbers, it's highly correlated.
    
    def get_score(n):
        if n < 2:
            return 0
        divs = []
        temp = n
        i = 2
        while i * i <= temp:
            if temp % i == 0:
                while temp % i == 0:
                    temp //= i
            i += 1 if i == 2 else 2
        # This factorization is too slow for 10^18 in Python if done repeatedly.
        # We will generate divisors from the prime factorization directly.
        pass

    # Better: Generate divisors from known factorization
    def score_from_factors(factors):
        # factors: list of (prime, exponent)
        # Generate all divisors
        divs = [1]
        for p, e in factors:
            new_divs = []
            for d in divs:
                val = d
                for _ in range(e):
                    new_divs.append(val)
                    val *= p
            divs = new_divs if not divs else divs + new_divs # Incorrect logic, need to expand
            # Correct expansion:
            current = divs
            divs = []
            for d in current:
                val = d
                for _ in range(e):
                    divs.append(val)
                    val *= p
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Search space for exponents of 2, 3, 5, 7, 11
    # Max exponent for 2 is around 20
    max_exp_2 = 20
    max_exp_3 = 15
    max_exp_5 = 10
    max_exp_7 = 8
    max_exp_11 = 6
    
    # We iterate through combinations
    # To save time, we prune if product > 10^18
    LIMIT = 10**18
    
    candidates = []
    
    for a in range(max_exp_2 + 1):
        p2 = 2**a
        if p2 * base_product > LIMIT:
            break
        for b in range(max_exp_3 + 1):
            p3 = 3**b
            if p2 * p3 * base_product > LIMIT:
                break
            for c in range(max_exp_5 + 1):
                p5 = 5**c
                if p2 * p3 * p5 * base_product > LIMIT:
                    break
                for d in range(max_exp_7 + 1):
                    p7 = 7**d
                    if p2 * p3 * p5 * p7 * base_product > LIMIT:
                        break
                    for e in range(max_exp_11 + 1):
                        p11 = 11**e
                        n_candidate = p2 * p3 * p5 * p7 * p11 * base_product
                        if n_candidate > LIMIT:
                            break
                        
                        # Check time
                        if time.time() > deadline - 10:
                            return best_n if best_n > 1 else 2
                        
                        # Heuristic: Divisor count
                        div_count = (a+1)*(b+1)*(c+1)*(d+1)*(e+1) * (2**10) # base has 10 primes
                        # Store candidate
                        candidates.append((div_count, n_candidate, [(2,a), (3,b), (5,c), (7,d), (11,e)] + [(p,1) for p in base_primes]))
                        
    # Sort by divisor count descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Evaluate top candidates for exact score
    evaluated = set()
    for div_count, n, factors in candidates:
        if n in evaluated:
            continue
        evaluated.add(n)
        
        if time.time() > deadline - 5:
            break
            
        s = score_from_factors(factors)
        if s > best_score:
            best_score = s
            best_n = n
            
    return best_n
