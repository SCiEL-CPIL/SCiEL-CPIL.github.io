import math
import time

def construct():
    start_time = time.time()
    time_limit = 140 # Safety margin for 150s limit
    
    max_val = 10**18
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    
    candidates = set()
    
    def dfs(idx, max_exp, current_n, current_tau):
        if time.time() - start_time > 60:
            return
            
        if current_n > 1:
            candidates.add((current_tau, current_n))
            
        if idx >= len(primes):
            return
            
        p = primes[idx]
        limit_n = max_val // current_n
        
        e_max = 0
        temp = 1
        while temp * p <= limit_n and e_max < max_exp:
            temp *= p
            e_max += 1
            
        for e in range(1, e_max + 1):
            next_n = current_n * (p ** e)
            next_tau = current_tau * (e + 1)
            dfs(idx + 1, e, next_n, next_tau)
            
            if time.time() - start_time > 60:
                return

    dfs(0, 60, 1, 1)
    
    # Add Primorials explicitly to ensure they are considered
    # Even if they are in candidates, we want to prioritize them in evaluation
    primorial_candidates = []
    pk = 1
    for i, p in enumerate(primes):
        pk *= p
        if pk > max_val:
            break
        primorial_candidates.append(pk)
        # Add to main candidates too
        candidates.add((2**(i+1), pk))
        
        # Variations
        for a in range(1, 5):
            n = pk * (2 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))
                
        for a in range(1, 4):
            n = pk * (3 ** a)
            if n <= max_val:
                candidates.add((2**(i+1) * (a+1), n))

    if not candidates:
        return 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47

    candidate_list = list(candidates)
    candidate_list.sort(key=lambda x: x[0], reverse=True)
    
    def tau_perp(n):
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
        gcd = math.gcd
        for i in range(len(divs) - 1):
            if gcd(divs[i], divs[i+1]) == 1:
                score += 1
        return score

    best_score = -1
    best_n = candidate_list[0][1]
    
    # Evaluate top 100 by tau
    eval_limit = 100
    evaluated = set()
    
    for i, (tau, n) in enumerate(candidate_list):
        if i >= eval_limit:
            break
        if time.time() - start_time > time_limit - 10:
            break
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    # Evaluate Primorials specifically
    # They might have lower tau but higher tau_perp ratio
    for n in primorial_candidates:
        if time.time() - start_time > time_limit - 5:
            break
        if n in evaluated:
            continue
        evaluated.add(n)
        
        score = tau_perp(n)
        if score > best_score:
            best_score = score
            best_n = n
            
    return best_n
