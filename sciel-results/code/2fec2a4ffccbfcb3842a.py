import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    MAX_N = 10**18
    
    # Generate primes up to a reasonable limit
    def primes_up_to(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = False
        return [i for i, v in enumerate(sieve) if v]
    
    # We need primes up to maybe 50-100 to get enough small primes
    PRIMES = primes_up_to(200)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def tau_perp_score(n):
        """Count adjacent divisor pairs with gcd 1."""
        if n < 2:
            return 0
        # Generate all divisors
        divisors = []
        # Factorize n
        temp = n
        factors = []
        for p in PRIMES:
            if p * p > temp:
                break
            if temp % p == 0:
                exp = 0
                while temp % p == 0:
                    temp //= p
                    exp += 1
                factors.append((p, exp))
        if temp > 1:
            factors.append((temp, 1))
        
        # Generate divisors
        divisors = [1]
        for p, e in factors:
            new_divs = []
            pe = 1
            for i in range(e + 1):
                for d in divisors:
                    new_divs.append(d * pe)
                pe *= p
            divisors = new_divs
        
        divisors.sort()
        count = 0
        for i in range(len(divisors) - 1):
            if gcd(divisors[i], divisors[i+1]) == 1:
                count += 1
        return count
    
    def score_from_exponents(exponents, primes_used):
        """Compute tau_perp score from exponent list and corresponding primes."""
        # Build n
        n = 1
        for i, e in enumerate(exponents):
            if e > 0:
                n *= primes_used[i] ** e
        if n > MAX_N:
            return -1
        return tau_perp_score(n)
    
    # Strategy: Use DFS with pruning to find optimal exponent distribution
    # Key insight: to maximize coprime adjacent pairs, we want many distinct prime factors
    # with exponents that create many divisors where consecutive divisors are coprime.
    
    best_score = 0
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19  # Primorial of first 8 primes
    
    def evaluate_candidate(exponents):
        nonlocal best_score, best_n
        n = 1
        for i, e in enumerate(exponents):
            if e > 0:
                n *= PRIMES[i] ** e
        if n > MAX_N or n < 2:
            return
        score = tau_perp_score(n)
        if score > best_score:
            best_score = score
            best_n = n
    
    # DFS over exponent assignments
    # Exponents should be non-increasing for optimality (standard result for maximizing divisors)
    # But here we're maximizing coprime adjacent pairs, which may favor different structures
    
    def dfs(idx, current_n, max_exp, current_exponents):
        nonlocal best_score, best_n
        if time.time() > deadline:
            return
        
        # Update best if valid
        if current_n >= 2 and current_n <= MAX_N:
            score = tau_perp_score(current_n)
            if score > best_score:
                best_score = score
                best_n = current_n
        
        if idx >= len(PRIMES):
            return
        
        # Try all possible exponents from 0 to max_exp
        for e in range(max_exp, -1, -1):
            new_n = current_n * (PRIMES[idx] ** e) if e > 0 else current_n
            if new_n > MAX_N:
                continue
            current_exponents.append(e)
            dfs(idx + 1, new_n, e, current_exponents)
            current_exponents.pop()
    
    # Start DFS with reasonable bounds
    # First prime can have high exponent, but let's cap it
    dfs(0, 1, 60, [])
    
    # Also try some heuristic candidates
    # Primorial of first k primes
    for k in range(2, 16):
        n = 1
        for i in range(k):
            n *= PRIMES[i]
            if n > MAX_N:
                break
        if n <= MAX_N:
            score = tau_perp_score(n)
            if score > best_score:
                best_score = score
                best_n = n
    
    # Try numbers with more exponents on smaller primes
    for e2 in range(1, 20):
        n = 2 ** e2
        if n > MAX_N:
            break
        for e3 in range(1, 15):
            n2 = n * 3 ** e3
            if n2 > MAX_N:
                break
            for e5 in range(1, 10):
                n3 = n2 * 5 ** e5
                if n3 > MAX_N:
                    break
                score = tau_perp_score(n3)
                if score > best_score:
                    best_score = score
                    best_n = n3
                if time.time() > deadline:
                    return best_n
    
    return best_n
