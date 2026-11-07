import time
import math
import sys
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    max_m = 10**18
    
    # Sieve for sigma(k)
    L = 10**6
    
    sigma = [0] * (L + 1)
    
    # Compute sum of divisors for all numbers up to L
    for d in range(1, L + 1):
        for m in range(d, L + 1, d):
            sigma[m] += d
            
    # Map M = k * sigma(k) to list of k
    solutions = defaultdict(list)
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            solutions[m_val].append(k)
            
    # Sort candidates by number of solutions, take top few
    candidates = sorted(solutions.items(), key=lambda x: -len(x[1]))
    
    best_m = 0
    best_k_list = []
    
    # For each top candidate, try to find additional k values
    for m_val, k_list in candidates[:50]:
        if len(k_list) <= len(best_k_list) and len(k_list) < 3:
            break
            
        # Try to find more k values by checking divisors of m_val
        # k must satisfy k*sigma(k) = m_val, so k divides m_val
        # and k <= sqrt(m_val)
        sqrt_m = math.isqrt(m_val)
        
        # Get all divisors of m_val up to sqrt_m
        # But m_val can be up to 10^18, so we need to factor it
        # Only do this for promising candidates
        if len(k_list) >= 3 and time.time() < deadline - 30:
            # Factor m_val to find divisors
            n = m_val
            divisors = [1]
            remaining = n
            p = 2
            while p * p <= remaining:
                if remaining % p == 0:
                    exp = 0
                    while remaining % p == 0:
                        remaining //= p
                        exp += 1
                    # Multiply existing divisors by p^0, p^1, ..., p^exp
                    new_divs = []
                    pow_p = 1
                    for i in range(exp + 1):
                        for d in divisors:
                            new_divs.append(d * pow_p)
                        pow_p *= p
                    divisors = new_divs
                p += 1 if p == 2 else 2
            if remaining > 1:
                new_divs = []
                for d in divisors:
                    new_divs.append(d)
                    new_divs.append(d * remaining)
                divisors = new_divs
            
            # Check each divisor k of m_val where k <= sqrt_m
            existing = set(k_list)
            for k in divisors:
                if k in existing or k > sqrt_m or k < 1:
                    continue
                # Check if k*sigma(k) == m_val
                # We need sigma(k), which we can compute by factoring k
                if k <= L:
                    sk = sigma[k]
                else:
                    # Compute sigma(k) by factoring
                    sk = _sigma(k)
                if k * sk == m_val:
                    existing.add(k)
                    k_list.append(k)
        
        if len(k_list) > len(best_k_list):
            best_m = m_val
            best_k_list = list(k_list)
    
    # Fallback if no solutions found
    if not best_k_list:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Ensure k_i <= sqrt(M)
    sqrt_m = math.isqrt(best_m)
    valid_k = sorted(set(k for k in best_k_list if k <= sqrt_m and k >= 1))
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

def _sigma(n):
    """Compute sigma(n) by trial division for n > L."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            power_sum = 1
            while n % d == 0:
                n //= d
                power_sum += d
            result *= power_sum
        d += 1 if d == 2 else 2
    if n > 1:
        result *= (1 + n)
    return result

if __name__ == "__main__":
    result = construct()
    print(result)
