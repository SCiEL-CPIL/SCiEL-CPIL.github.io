import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 148  # Tighter safety margin below 150s
    
    # Primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    max_val = 10**18
    
    candidates = []
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > TIME_LIMIT - 5: # Reserve 5s for evaluation
            return
            
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        # Max exponent for this prime
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
            
        # Iterate exponents
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
    # Start DFS
    dfs(0, 60, 1, 1)
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    # Evaluate tau_perp for all candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_n = candidates[0][1]
    best_score = -1
    
    def tau_perp(n):
        # Factorize n using primes list (since n is constructed from them)
        divs = [1]
        temp = n
        p_idx = 0
        while temp > 1 and p_idx < len(primes):
            p = primes[p_idx]
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                current_size = len(divs)
                factor = 1
                for _ in range(count):
                    factor *= p
                    for i in range(current_size):
                        divs.append(divs[i] * factor)
            p_idx += 1
            
        divs.sort()
        score = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    # Evaluate top candidates first
    for tau, n in candidates:
        if time.time() - start_time > TIME_LIMIT - 2:
            break
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Local search: try adjusting exponents of best_n
    if time.time() - start_time < TIME_LIMIT - 10:
        # Get factorization of best_n
        best_factors = {}
        temp = best_n
        for p in primes:
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                best_factors[p] = count
            if temp == 1:
                break
        
        # Try local moves: increase/decrease exponents by 1
        local_candidates = []
        for p in list(best_factors.keys()):
            e = best_factors[p]
            # Try increasing
            new_n = best_n * p
            if new_n <= max_val:
                local_candidates.append(new_n)
            # Try decreasing
            if e > 1:
                new_n = best_n // p
                if new_n >= 2:
                    local_candidates.append(new_n)
        
        # Also try adding new small primes if room
        for p in primes:
            if p not in best_factors:
                new_n = best_n * p
                if new_n <= max_val:
                    local_candidates.append(new_n)
        
        # Evaluate local candidates
        for n in local_candidates:
            if time.time() - start_time > TIME_LIMIT - 3:
                break
            score = tau_perp(n)
            if score > best_score:
                best_score = score
                best_n = n
                
    return best_n
