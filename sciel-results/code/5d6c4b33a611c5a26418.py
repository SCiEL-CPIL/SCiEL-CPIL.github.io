"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s limit
    
    LIMIT = 10**18
    
    # Precompute primes up to 200
    primes = []
    for x in range(2, 200):
        is_prime = True
        for p in primes:
            if p * p > x:
                break
            if x % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(x)
    
    def get_divisors_from_factors(factors):
        """Generate all divisors from list of (prime, exponent) pairs."""
        divisors = [1]
        for p, e in factors:
            new_divs = []
            for d in divisors:
                val = d
                for _ in range(e + 1):
                    new_divs.append(val)
                    val *= p
            divisors = new_divs
        divisors.sort()
        return divisors
    
    def compute_score(divisors):
        """Compute tau_perp score from sorted divisors."""
        score = 0
        for i in range(len(divisors) - 1):
            if math.gcd(divisors[i], divisors[i + 1]) == 1:
                score += 1
        return score
    
    def score_factors(factors):
        """Score a factorization: returns (score, n) or (-1, 0) if n > LIMIT."""
        n = 1
        for p, e in factors:
            n *= p ** e
            if n > LIMIT:
                return -1, 0
        if n < 2:
            return -1, 0
        divisors = get_divisors_from_factors(factors)
        s = compute_score(divisors)
        return s, n
    
    # Key insight: we want to maximize coprime adjacent pairs.
    # For squarefree n with k primes, the divisors are all 2^k subsets.
    # The number of divisors is 2^k, and we get 2^k - 1 adjacent pairs.
    # Many of these will be coprime.
    
    # Strategy: 
    # 1. First, find the best squarefree number (all exponents = 1)
    # 2. Then try replacing some primes with higher exponents on smaller primes
    # 3. Use a smarter search: try different combinations of primes and exponents
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # fallback
    
    # Step 1: Try squarefree with k smallest primes for k = 2..15
    for k in range(2, 16):
        if time.time() > DEADLINE - 5:
            break
        factors = [(primes[i], 1) for i in range(k)]
        s, n = score_factors(factors)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Step 2: Try replacing the largest prime with a square of a smaller prime
    # For k primes, try: use primes[0..k-2] with exponent 1, and primes[j]^2 for some j
    # Or: use primes[0..k-3] with exponent 1, and two primes with exponent 2
    
    # More systematic: for k = 10..14, try various exponent patterns
    # where most exponents are 1 but a few are 2 or 3
    
    def search_with_exponents(k, max_exp=3, deadline=DEADLINE):
        """Search over exponent patterns for the k smallest primes."""
        nonlocal best_score, best_n
        if time.time() > deadline:
            return
        
        # Recursive search with pruning
        def recurse(idx, current_n, current_exps):
            nonlocal best_score, best_n
            if time.time() > deadline:
                return
            
            if idx == k:
                factors = [(primes[i], current_exps[i]) for i in range(k)]
                s, n = score_factors(factors)
                if s > best_score:
                    best_score = s
                    best_n = n
                return
            
            p = primes[idx]
            for e in range(1, max_exp + 1):
                new_n = current_n * (p ** e)
                if new_n > LIMIT:
                    break
                current_exps[idx] = e
                recurse(idx + 1, new_n, current_exps)
        
        current_exps = [1] * k
        recurse(0, 1, current_exps)
    
    # Search for k = 10 to 14 with exponents up to 3
    for k in range(10, 15):
        if time.time() > DEADLINE - 15:
            break
        search_with_exponents(k, max_exp=3)
    
    # Step 3: Try k = 9 with higher exponents (up to 4)
    for k in range(8, 10):
        if time.time() > DEADLINE - 15:
            break
        search_with_exponents(k, max_exp=4)
    
    # Step 4: Try non-consecutive prime sets
    # For k primes, we might skip some small primes and use larger ones
    # to allow higher exponents on the small primes
    
    # For example: for k=12, try using primes [2,3,5,7,11,13,17,19,23,29,31,37]
    # vs [2,3,5,7,11,13,17,19,23,29,31,41] (skip 37, use 41)
    # The latter might allow 2^2 if the product is smaller
    
    # Actually, the key insight is: if we use a larger prime instead of a smaller one,
    # the product increases, so we can't afford higher exponents.
    # So using the smallest primes is optimal for fitting more primes.
    
    # Step 5: Try a different approach - use fewer primes but with higher exponents
    # to create more divisors that are coprime to their neighbors
    
    # For k=7 primes with some exponents=2:
    # n = 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
    # where a,b,c,d,e,f,g >= 1
    
    # The number of divisors is (a+1)(b+1)...(g+1)
    # We want this to be large, but n <= 10^18
    
    # Let's try: 2^3 * 3^3 * 5^3 * 7^2 * 11^2 * 13^2 * 17 * 19
    # = 8 * 27 * 125 * 49 * 121 * 169 * 17 * 19
    # Let me compute this...
    
    # Actually, let's just do a broader search
    # For k = 7 to 11, try exponents 1 to 5
    
    for k in range(7, 12):
        if time.time() > DEADLINE - 15:
            break
        search_with_exponents(k, max_exp=5)
    
    # Step 6: Try a completely different structure
    # Use many small primes with exponent 1, but replace the largest one
    # with a product of two primes (i.e., use k+1 primes but one has exponent 0)
    # Wait, that's the same as using k primes.
    
    # Actually, the key is: for a given number of divisors, we want the divisors
    # to be "spread out" so that adjacent ones are likely coprime.
    
    # Let's try: n = 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    # This has 13 prime factors (one with exponent 2), so 2*1^12 = 2... wait
    # Number of divisors = 2 * 1^12 = 2? No, (2+1)*1^12 = 3
    # That's not right. Let me reconsider.
    
    # n = 2^2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    # tau(n) = (2+1) * 1^12 = 3? No, that's wrong.
    # tau(n) = (2+1) * (1+1)^12 = 3 * 4096 = 12288
    # n = 4 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41
    # = 4 * (3*5*7*11*13*17*19*23*29*31*37*41)
    # 3*5*7*11*13*17*19*23*29*31*37*41 = let me compute
    # 3*5=15, *7=105, *11=1155, *13=15015, *17=255255, *19=4849845,
    # *23=111546435, *29=3234846615, *31=100280245065, *37=3710369067405,
    # *41=152125131763605
    # So n = 4 * 152125131763605 = 608500527054420 < 10^18. Good.
    
    # Let's try this and similar constructions
    test_constructions = [
        # (exponents for primes[0..k-1])
        [2,1,1,1,1,1,1,1,1,1,1,1,1],  # 2^2 * 3 * ... * 41
        [2,2,1,1,1,1,1,1,1,1,1,1],    # 2^2 * 3^2 * 5 * ... * 37
        [2,2,2,1,1,1,1,1,1,1,1],      # 2^2 * 3^2 * 5^2 * 7 * ... * 31
        [3,1,1,1,1,1,1,1,1,1,1,1],    # 2^3 * 3 * ... * 37
        [3,2,1,1,1,1,1,1,1,1,1],      # 2^3 * 3^2 * 5 * ... * 31
        [2,1,1,1,1,1,1,1,1,1,1,1,1,1], # 2^2 * 3 * ... * 43
    ]
    
    for exps in test_constructions:
        if time.time() > DEADLINE - 5:
            break
        k = len(exps)
        factors = [(primes[i], exps[i]) for i in range(k)]
        s, n = score_factors(factors)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Step 7: Try a greedy approach - start with the best squarefree and try
    # increasing exponents one at a time, keeping the best
    
    # Start with the best squarefree found
    # Find the best squarefree
    best_sq_score = 0
    best_sq_k = 0
    for k in range(2, 16):
        factors = [(primes[i], 1) for i in range(k)]
        s, n = score_factors(factors)
        if s > best_sq_score:
            best_sq_score = s
            best_sq_k = k
    
    # Now try increasing exponents on the best squarefree
    if best_sq_k >= 2:
        base_exps = [1] * best_sq_k
        for i in range(best_sq_k):
            for e in range(2, 5):
                if time.time() > DEADLINE - 5:
                    break
                exps = base_exps.copy()
                exps[i] = e
                factors = [(primes[j], exps[j]) for j in range(best_sq_k)]
                s, n = score_factors(factors)
                if s > best_score:
                    best_score = s
                    best_n = n
    
    # Step 8: Try a local search from the best found so far
    # Represent the current best as a list of (prime, exponent)
    # Try changing one exponent at a time
    
    # First, find the factorization of best_n
    def factorize(n):
        factors = []
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors.append((p, e))
        if temp > 1:
            factors.append((temp, 1))
        return factors
    
    best_factors = factorize(best_n)
    
    # Local search: try changing each exponent
    for iteration in range(50):
        if time.time() > DEADLINE - 5:
            break
        improved = False
        for i in range(len(best_factors)):
            p, e = best_factors[i]
            for new_e in range(1, e + 3):
                if new_e == e:
                    continue
                new_factors = best_factors.copy()
                new_factors[i] = (p, new_e)
                s, n = score_factors(new_factors)
                if s > best_score:
                    best_score = s
                    best_n = n
                    best_factors = new_factors
                    improved = True
                    break
        if not improved:
            break
    
    # Final validation
    if best_n < 2 or best_n > LIMIT:
        best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    
    return best_n

# EVOLVE-BLOCK-END
