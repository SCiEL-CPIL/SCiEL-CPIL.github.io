import math
import time

def construct():
    start_time = time.time()
    time_limit = 110 # Safety margin for 120s limit
    
    max_val = 10**18
    # Primes up to 47. Product of first 15 primes is ~6.1e17.
    # This covers the majority of high-tau candidates.
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    candidates = [] # List of (tau, n)
    
    # DFS to generate candidates with non-increasing exponents for a prefix of primes.
    # We do NOT allow skipping primes in the middle (e.g., 2*5 without 3) to keep search space small.
    # This assumes optimal n uses the smallest primes.
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 50: # Stop generating after 50s
            return
            
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Calculate max exponent for this prime
        e_max = 0
        temp = 1
        # Ensure e <= max_exp
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
            
        # Try exponents 1 to e_max.
        # Recursing implies using the next prime.
        # Not recursing (stopping) is handled by the fact that current_n is a valid candidate.
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 50:
                return

    # Start DFS
    # Max exponent for 2 is ~59
    dfs(0, 60, 1, 1)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Sort by tau descending. Higher tau correlates with higher potential tau_perp.
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    def tau_perp(n):
        # Factorize n using small primes
        factors = []
        temp_n = n
        for p in primes:
            if temp_n % p == 0:
                count = 0
                while temp_n % p == 0:
                    temp_n //= p
                    count += 1
                factors.append((p, count))
            if temp_n == 1:
                break
        
        if temp_n > 1:
            factors.append((temp_n, 1))

        # Generate divisors
        divs = [1]
        for p, e in factors:
            current_divs = list(divs)
            factor = 1
            for _ in range(e):
                factor *= p
                for d in current_divs:
                    divs.append(d * factor)
                    
        divs.sort()
        
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = -1
    best_n = candidates[0][1]
    
    # Evaluate top candidates
    # Limit to top 1000 to ensure speed
    eval_limit = 1000
    
    for i, (tau, n) in enumerate(candidates):
        if i >= eval_limit:
            break
        if time.time() - start_time > time_limit - 10:
            break
            
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
