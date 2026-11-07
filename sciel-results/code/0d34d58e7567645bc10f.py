import time
import math
import sys
from array import array
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    max_m = 10**18
    
    # Phase 1: Sieve-based search
    # Use a moderate L to find initial candidates
    L = 2 * 10**6
    
    # Sieve for sigma(k) using array for memory efficiency
    sigma = array('q', [0]) * (L + 1)
    
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
            
    # Find top candidates by number of solutions
    candidates = []
    for m, k_list in solutions.items():
        if len(k_list) >= 2:
            candidates.append((len(k_list), m, k_list))
    
    # Sort by count descending, then by M ascending for determinism
    candidates.sort(key=lambda x: (-x[0], x[1]))
    
    best_count = 0
    best_m = 0
    best_k_list = []
    
    # Phase 2: Targeted extension for top candidates
    # For each top candidate, try to find additional k values
    # by exploring the prime factorization structure
    
    def factorize(n):
        """Return prime factorization of n as dict {prime: exponent}"""
        factors = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                factors[d] = factors.get(d, 0) + 1
                n //= d
            d += 1
        if n > 1:
            factors[n] = factors.get(n, 0) + 1
        return factors
    
    def sigma_from_factors(factors):
        """Compute sigma from prime factorization"""
        result = 1
        for p, e in factors.items():
            result *= (p**(e+1) - 1) // (p - 1)
        return result
    
    def k_from_factors(factors):
        """Compute k from prime factorization"""
        result = 1
        for p, e in factors.items():
            result *= p**e
        return result
    
    # Process top candidates for extension
    for count, m, k_list in candidates[:50]:  # Top 50 candidates
        if time.time() > deadline:
            break
            
        if count <= best_count:
            break
            
        # Get the set of primes in the known solutions
        known_primes = set()
        for k in k_list:
            fac = factorize(k)
            known_primes.update(fac.keys())
        
        # Try to find additional k values with the same prime support
        # that satisfy k * sigma(k) = m
        additional_k = set()
        
        # For each known k, explore nearby exponent combinations
        for k in k_list:
            if time.time() > deadline:
                    break
            fac = factorize(k)
            # Try small variations in exponents
            for p in list(fac.keys()):
                for delta_e in [-2, -1, 1, 2]:
                    new_e = fac[p] + delta_e
                    if new_e < 0:
                        continue
                    new_fac = dict(fac)
                    new_fac[p] = new_e
                    if new_e == 0:
                        del new_fac[p]
                    new_k = k_from_factors(new_fac)
                    if new_k <= 1:
                        continue
                    new_sigma = sigma_from_factors(new_fac)
                    if new_k * new_sigma == m:
                        additional_k.add(new_k)
        
        # Also try: for each prime in known_primes, try different exponent combinations
        # This is more expensive, so only do it for the most promising candidates
        if count >= 3:
            # Generate candidate exponent combinations
            primes_list = sorted(known_primes)
            # For each prime, try exponents in a small range
            # around what we'd expect from k ~ sqrt(m)
            sqrt_m = math.isqrt(m)
            
            def search_exponents(idx, current_k, current_sigma, current_fac):
                if time.time() > deadline:
                    return []
                
                if idx == len(primes_list):
                    if current_k > 1 and current_k * current_sigma == m:
                        return [current_k]
                    return []
                
                p = primes_list[idx]
                results = []
                
                # Determine reasonable exponent range
                # k should be roughly in range [1, sqrt_m]
                # Try exponents from 0 to a reasonable max
                max_exp = 0
                temp = 1
                while temp * p <= sqrt_m * 10:  # generous upper bound
                    temp *= p
                    max_exp += 1
                max_exp = min(max_exp, 30)
                
                for e in range(max_exp + 1):
                    if e == 0:
                        new_k = current_k
                        new_sigma = current_sigma
                        new_fac = dict(current_fac)
                    else:
                        p_power = p**e
                        sum_powers = (p**(e+1) - 1) // (p - 1)
                        new_k = current_k * p_power
                        new_sigma = current_sigma * sum_powers
                        new_fac = dict(current_fac)
                        new_fac[p] = e
                    
                    if new_k > sqrt_m:
                        break  # Further exponents will only make k larger
                    
                    results.extend(search_exponents(idx + 1, new_k, new_sigma, new_fac))
                
                return results
            
            try:
                extra_k = search_exponents(0, 1, 1, {})
                additional_k.update(extra_k)
            except Exception:
                pass
        
        # Combine known and additional
        all_k = set(k_list) | additional_k
        all_k = [k for k in all_k if k <= math.isqrt(m)]
        
        if len(all_k) > best_count:
            best_count = len(all_k)
            best_m = m
            best_k_list = sorted(all_k)
    
    # If we found something better, use it; otherwise use the sieve result
    if best_count > 0 and best_m > 0:
        valid_k = [k for k in best_k_list if k <= math.isqrt(best_m)]
        valid_k.sort()
        M_out = int(best_m)
        k_out = [int(k) for k in valid_k]
        return {'M': M_out, 'k': k_out}
    
    # Fallback: use the best from sieve
    if candidates:
        count, m, k_list = candidates[0]
        valid_k = [k for k in k_list if k <= math.isqrt(m)]
        valid_k.sort()
        M_out = int(m)
        k_out = [int(k) for k in valid_k]
        return {'M': M_out, 'k': k_out}
    
    # Ultimate fallback
    return {'M': 333312, 'k': [336, 372, 434]}

if __name__ == "__main__":
    result = construct()
    print(result)
