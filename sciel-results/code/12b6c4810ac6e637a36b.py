import math
import time

def construct():
    """
    Returns an integer n <= 10^18 maximizing the number of adjacent
    coprime pairs in its divisor sequence.
    """
    start_time = time.time()
    TIME_LIMIT = 145  # Safety margin below 150s
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71]
    
    candidates = []
    
    # DFS to find numbers with high divisor counts (tau)
    # We store (tau, n)
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > TIME_LIMIT:
            return
        
        if current_n > 1:
            candidates.append((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max_possible = 0
        temp = 1
        while temp * p <= limit_n and e_max_possible < max_exp:
            temp *= p
            e_max_possible += 1
            
        for e in range(1, e_max_possible + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > TIME_LIMIT:
                return

    # Start DFS
    dfs(0, 60, 1, 1)
    
    # Also add the square-free number with maximum primes
    sf_n = 1
    for p in primes:
        if sf_n * p <= max_val:
            sf_n *= p
        else:
            break
    # Calculate tau for sf_n (it is 2^k)
    k = len([p for p in primes if sf_n % p == 0])
    candidates.append((2**k, sf_n))
    
    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
        
    # Sort by tau descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    best_n = candidates[0][1]
    best_score = -1
    
    def tau_perp(n):
        # Factorize n using the primes list
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

    # Evaluate top candidates
    # The number of candidates might be large, so limit evaluation
    # But sorting is O(N log N). N is likely < 10000.
    # We evaluate top 100 by tau.
    eval_count = min(len(candidates), 100)
    
    for tau, n in candidates[:eval_count]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
