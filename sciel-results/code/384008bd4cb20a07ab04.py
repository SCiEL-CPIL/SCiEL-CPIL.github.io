import math
import time
import numpy as np
from collections import defaultdict

def is_prime(n):
    if n < 2: return False
    if n == 2 or n == 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def factorize(n):
    factors = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append(n)
    return factors

def sigma_from_factors(factors):
    # factors is a list of prime factors (with multiplicity)
    # Group them
    prime_powers = {}
    for p in factors:
        prime_powers[p] = prime_powers.get(p, 0) + 1
    
    sigma_val = 1
    for p, e in prime_powers.items():
        # sum_{i=0}^e p^i = (p^(e+1)-1)/(p-1)
        sigma_val *= (p**(e+1) - 1) // (p - 1)
    return sigma_val

def get_divisors(n):
    factors = factorize(n)
    # Generate divisors from prime factorization
    # This is a bit complex, let's use a simpler method for small n or just trial division for sqrt(n) if n is small enough?
    # For n up to 10^18, trial division up to 10^9 is too slow.
    # We need efficient divisor generation.
    
    # Let's refine factorize to return {p: e}
    def get_prime_factorization(n):
        pf = {}
        d = 2
        while d * d <= n:
            while n % d == 0:
                pf[d] = pf.get(d, 0) + 1
                n //= d
            d += 1 if d == 2 else 2
        if n > 1:
            pf[n] = pf.get(n, 0) + 1
        return pf

    pf = get_prime_factorization(n)
    divisors = [1]
    for p, e in pf.items():
        new_divs = []
        power = 1
        for _ in range(e + 1):
            for d in divisors:
                new_divs.append(d * power)
            power *= p
        divisors = new_divs
    return divisors

def construct():
    start_time = time.time()
    deadline = 110
    
    best_M = 333312
    best_k_list = [336, 372, 434]
    best_count = 3
    
    target_ns = [5_000_000, 10_000_000]
    
    # We will store top candidates
    top_candidates = {} # M -> count
    
    for N in target_ns:
        if time.time() > deadline - 30: # Leave 30s for verification
            break
            
        try:
            sigma = np.zeros(N + 1, dtype=np.int64)
            for d in range(1, N + 1):
                sigma[d::d] += d
                
            k_vals = np.arange(1, N + 1, dtype=np.int64)
            M_vals = k_vals * sigma
            
            mask = M_vals <= 10**18
            M_valid = M_vals[mask]
            k_valid = k_vals[mask]
            
            if len(M_valid) == 0:
                continue
                
            unique_M, counts, inverse = np.unique(M_valid, return_counts=True, return_inverse=True)
            
            # Get top 10 indices
            if len(counts) > 10:
                top_idx = np.argsort(counts)[-10:][::-1]
            else:
                top_idx = np.arange(len(counts))
                
            for idx in top_idx:
                m_val = int(unique_M[idx])
                c_val = int(counts[idx])
                
                # Update top candidates
                if m_val not in top_candidates or c_val > top_candidates[m_val]:
                    top_candidates[m_val] = c_val
                    
            # Update best from this batch immediately
            max_idx = np.argmax(counts)
            current_count = int(counts[max_idx])
            current_M = int(unique_M[max_idx])
            
            if current_count > best_count:
                best_count = current_count
                best_M = current_M
                k_indices = np.where(inverse == max_idx)[0]
                best_k_list = [int(x) for x in k_valid[k_indices]]
                
        except MemoryError:
            break
        except Exception:
            break
            
    # Verification phase for top candidates
    # We want to find if any candidate M has more solutions than best_count
    # Check the top 5 candidates from top_candidates
    
    # Sort top_candidates by count descending
    sorted_candidates = sorted(top_candidates.items(), key=lambda x: x[1], reverse=True)
    
    for M, count_in_sieve in sorted_candidates[:5]:
        if time.time() > deadline:
            break
            
        # If the sieve count is already less than best_count, it can't be better 
        # UNLESS we find more solutions outside the sieve range.
        # But if M is small, sieve covered everything.
        # If M is large, sieve missed large k.
        
        # Let's verify M
        # We need to find all k such that k * sigma(k) = M
        # k must be a divisor of M
        # and k <= sqrt(M)
        
        divisors = get_divisors(M)
        valid_ks = []
        sqrt_M = math.isqrt(M)
        
        for k in divisors:
            if k > sqrt_M:
                continue
            # Check if k * sigma(k) == M
            # Compute sigma(k)
            # We can use the sieve if k <= N_last, otherwise factorize
            # Since we don't have global sigma array, we factorize k
            # This is fast for small k
            try:
                if k <= 10_000_000:
                    # We could re-use sieve logic, but we don't have it stored.
                    # Just factorize.
                    pass
                sigma_k = sigma_from_factors(factorize(k))
                if k * sigma_k == M:
                    valid_ks.append(k)
            except Exception:
                continue
                
        if len(valid_ks) > best_count:
            best_count = len(valid_ks)
            best_M = M
            best_k_list = valid_ks
            
    return {'M': int(best_M), 'k': best_k_list}
