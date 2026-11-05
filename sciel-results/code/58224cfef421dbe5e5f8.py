def construct():
    import math
    import time
    import heapq
    from itertools import product

    start_time = time.time()
    deadline = start_time + 115.0

    # Primes up to ~100
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    LIMIT = 10**18
    
    def calc_score(n):
        if n < 2: return 0
        # Generate divisors
        divs = [1]
        temp = n
        i = 0
        # Factorize using the primes list
        while i < len(primes) and primes[i] * primes[i] <= temp:
            p = primes[i]
            if temp % p == 0:
                new_divs = []
                power = 1
                while temp % p == 0:
                    temp //= p
                    power *= p
                    for d in divs:
                        new_divs.append(d * power)
                divs += new_divs
            i += 1
        if temp > 1:
            for d in divs[:]:
                divs.append(d * temp)
        
        divs.sort()
        
        score = 0
        # Optimization: gcd(a,b) for adjacent divisors
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score

    def calc_score_fast(n, factorization):
        """Calculate score using pre-computed factorization."""
        if n < 2: return 0
        # Generate divisors from factorization
        divs = [1]
        for p, exp in factorization.items():
            if exp == 0:
                continue
            new_divs = []
            power = 1
            for _ in range(exp):
                power *= p
                for d in divs:
                    new_divs.append(d * power)
            divs += new_divs
        
        divs.sort()
        
        score = 0
        for j in range(len(divs) - 1):
            if math.gcd(divs[j], divs[j+1]) == 1:
                score += 1
        return score

    # First, evaluate the primorial of first 15 primes as a strong baseline
    primorial = 1
    primorial_factorization = {}
    for i, p in enumerate(primes[:15]):
        primorial *= p
        primorial_factorization[p] = 1
    
    best_n = primorial
    best_score = calc_score(primorial)
    best_factorization = dict(primorial_factorization)
    
    # Local search: try varying exponents of small primes
    # Focus on primes 2-13 (first 6 primes) with exponents 0-4
    # This explores the neighborhood of the primorial
    
    small_primes = primes[:6]  # 2, 3, 5, 7, 11, 13
    max_exp = 4
    other_primes = primes[6:15]  # Keep these at exponent 1
    
    # Calculate base product for other primes
    base_product = 1
    base_factorization = {}
    for p in other_primes:
        base_product *= p
        base_factorization[p] = 1
    
    # Search over exponent combinations for small primes
    # Use a priority queue to explore most promising combinations first
    # Priority based on estimated score (higher divisor count = potentially more coprime pairs)
    
    candidates = []
    K = 500
    
    def estimate_score(factorization):
        """Estimate score based on divisor count and structure."""
        tau = 1
        for exp in factorization.values():
            tau *= (exp + 1)
        # Heuristic: score is roughly proportional to sqrt(tau) for random-like distributions
        return tau ** 0.7
    
    # Generate initial candidates with varying exponents
    for exps in product(range(max_exp + 1), repeat=len(small_primes)):
        if time.time() > deadline - 30.0:
            break
        
        factorization = {}
        n = base_product
        valid = True
        
        for i, p in enumerate(small_primes):
            exp = exps[i]
            if exp > 0:
                factorization[p] = exp
                for _ in range(exp):
                    if n > LIMIT // p:
                        valid = False
                        break
                    n *= p
                if not valid:
                    break
        
        if not valid or n < 2 or n > LIMIT:
            continue
        
        # Add other primes
        for p, exp in base_factorization.items():
            factorization[p] = exp
        
        est = estimate_score(factorization)
        
        if len(candidates) < K:
            heapq.heappush(candidates, (-est, n, factorization))
        else:
            if est > -candidates[0][0]:
                heapq.heapreplace(candidates, (-est, n, factorization))
    
    # Evaluate top candidates
    seen = set()
    seen.add(best_n)
    
    while candidates:
        if time.time() > deadline - 10.0:
            break
        
        neg_est, n, factorization = heapq.heappop(candidates)
        
        if n in seen:
            continue
        seen.add(n)
        
        s = calc_score_fast(n, factorization)
        if s > best_score:
            best_score = s
            best_n = n
            best_factorization = dict(factorization)
    
    # Additional targeted search: try high exponents for 2 and 3
    # These often contribute significantly to coprime adjacent pairs
    
    for e2 in range(0, 11):
        for e3 in range(0, 7):
            if time.time() > deadline - 5.0:
                break
            
            n = base_product
            factorization = dict(base_factorization)
            
            # Add 2^e2
            if e2 > 0:
                factorization[2] = e2
                for _ in range(e2):
                    if n > LIMIT // 2:
                        break
                    n *= 2
            
            # Add 3^e3
            if e3 > 0:
                factorization[3] = e3
                for _ in range(e3):
                    if n > LIMIT // 3:
                        break
                    n *= 3
            
            if n < 2 or n > LIMIT:
                continue
            
            if n in seen:
                continue
            seen.add(n)
            
            s = calc_score_fast(n, factorization)
            if s > best_score:
                best_score = s
                best_n = n
                best_factorization = dict(factorization)
    
    # Final: try a few more variations with prime 17
    for e17 in range(0, 3):
        if time.time() > deadline - 2.0:
            break
        
        n = best_n
        # Remove 17 if present
        while n % 17 == 0:
            n //= 17
        
        # Add 17^e17
        for _ in range(e17):
            if n > LIMIT // 17:
                break
            n *= 17
        
        if n < 2 or n > LIMIT:
            continue
        
        if n in seen:
            continue
        seen.add(n)
        
        # Recalculate factorization
        temp = n
        factorization = {}
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factorization[p] = exp
        if temp > 1:
            factorization[temp] = 1
        
        s = calc_score_fast(n, factorization)
        if s > best_score:
            best_score = s
            best_n = n
    
    return best_n
