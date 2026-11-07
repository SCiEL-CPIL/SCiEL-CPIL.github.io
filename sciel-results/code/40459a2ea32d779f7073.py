import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s limit
    
    max_m = 10**18
    
    # Linear sieve for sigma(n) up to L
    L = 10**7
    
    sigma = [0] * (L + 1)
    lp = [0] * (L + 1)  # least prime factor
    spow = [1] * (L + 1)  # highest power of lp[i] dividing i
    primes = []
    
    sigma[1] = 1
    lp[1] = 1
    spow[1] = 1
    
    for i in range(2, L + 1):
        if lp[i] == 0:
            lp[i] = i
            primes.append(i)
            sigma[i] = 1 + i
            spow[i] = i
        for p in primes:
            if p > lp[i] or i * p > L:
                break
            lp[i * p] = p
            if i % p == 0:
                # p divides i, so spow[i*p] = spow[i] * p
                spow[i * p] = spow[i] * p
                # sigma(i*p) = sigma(i) - sigma(i / spow[i]) * (spow[i] + 1) + (spow[i]*p + 1) * sigma(i / spow[i])
                # More precisely: let i = p^e * m, gcd(m,p)=1
                # spow[i] = p^e
                # sigma(i) = sigma(m) * (1 + p + ... + p^e)
                # sigma(i*p) = sigma(m) * (1 + p + ... + p^(e+1))
                # = sigma(i) + sigma(m) * p^(e+1)
                # sigma(m) = sigma(i) / (1 + p + ... + p^e)
                # But we can use: sigma(i*p) = sigma(i) + spow[i]*p * sigma(i // spow[i])
                m_part = i // spow[i]
                sigma[i * p] = sigma[i] + spow[i] * p * sigma[m_part]
            else:
                # p does not divide i
                spow[i * p] = p
                sigma[i * p] = sigma[i] * (1 + p)
    
    # Map M = k * sigma(k) to list of k
    solutions = defaultdict(list)
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            solutions[m_val].append(k)
    
    # Sort candidates by number of solutions
    candidates = sorted(solutions.items(), key=lambda x: -len(x[1]))
    
    best_m = 0
    best_k_list = []
    
    # Helper: compute sigma(n) by trial division for n > L
    def _sigma_trial(n):
        if n <= 0:
            return 0
        result = 1
        d = 2
        while d * d <= n:
            if n % d == 0:
                power_sum = 1
                power = 1
                while n % d == 0:
                    n //= d
                    power *= d
                    power_sum += power
                result *= power_sum
            d += 1 if d == 2 else 2
        if n > 1:
            result *= (1 + n)
        return result
    
    # Helper: get all divisors of n
    def _divisors(n):
        divs = [1]
        d = 2
        while d * d <= n:
            if n % d == 0:
                exp = 0
                while n % d == 0:
                    n //= d
                    exp += 1
                new_divs = []
                pow_d = 1
                for _ in range(exp + 1):
                    for dv in divs:
                        new_divs.append(dv * pow_d)
                    pow_d *= d
                divs = new_divs
            d += 1 if d == 2 else 2
        if n > 1:
            new_divs = []
            for dv in divs:
                new_divs.append(dv)
                new_divs.append(dv * n)
            divs = new_divs
        return divs
    
    # For each top candidate, try to find additional k values by checking all divisors
    for m_val, k_list in candidates[:200]:
        if time.time() > deadline - 5:
            break
        
        if len(k_list) <= len(best_k_list) and len(k_list) < 4:
            break
        
        sqrt_m = math.isqrt(m_val)
        
        # Get all divisors of m_val up to sqrt_m
        divs = _divisors(m_val)
        
        existing = set(k_list)
        found_new = False
        
        for k in divs:
            if k in existing or k > sqrt_m or k < 1:
                continue
            # Check if k*sigma(k) == m_val
            if k <= L:
                sk = sigma[k]
            else:
                sk = _sigma_trial(k)
            if k * sk == m_val:
                existing.add(k)
                k_list.append(k)
                found_new = True
        
        if len(k_list) > len(best_k_list):
            best_m = m_val
            best_k_list = list(k_list)
    
    # Fallback if no solutions found
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Ensure k_i <= sqrt(M) and distinct
    sqrt_m = math.isqrt(best_m)
    valid_k = sorted(set(k for k in best_k_list if k <= sqrt_m and k >= 1))
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
