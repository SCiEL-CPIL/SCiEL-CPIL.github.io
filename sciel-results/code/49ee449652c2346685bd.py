"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sigma_by_factorization(n):
    """Compute sigma(n) by factoring n."""
    if n <= 0:
        return 0
    result = 1
    d = 2
    temp = n
    while d * d <= temp:
        if temp % d == 0:
            power = 0
            while temp % d == 0:
                temp //= d
                power += 1
            factor = 1
            d_power = 1
            for _ in range(power):
                d_power *= d
                factor += d_power
            result *= factor
        d += 1
    if temp > 1:
        result *= (1 + temp)
    return result

def _isqrt(x):
    return int(x ** 0.5)

def construct():
    deadline = time.time() + 140  # 140s deadline, 10s safety margin
    
    # Strategy: 
    # 1) Sieve sigma up to some N, find M with most k*sigma(k) collisions
    # 2) For the best M, search for additional k > N that divide M and satisfy k*sigma(k)=M
    # 3) Also try multiple candidate M's if time permits
    
    N = 5000000  # 5 million - larger sieve for more collisions
    
    # Compute sigma via sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
    
    # Group k by M_k = k * sigma(k)
    m_to_k = defaultdict(list)
    
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            m_to_k[M_val].append(k)
    
    # Find top candidate M's by number of solutions
    # Sort by count descending, take top few
    candidates = []
    for M_val, ks in m_to_k.items():
        if len(ks) >= 2:  # at least 2 solutions to be interesting
            candidates.append((len(ks), M_val, ks))
    
    candidates.sort(key=lambda x: -x[0])
    
    # Take top 10 candidates to try extending
    top_candidates = candidates[:10]
    
    best_M = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for count, M_val, ks in top_candidates:
        if time.time() > deadline - 30:
            break
        
        # Verify all k <= sqrt(M_val)
        sqrt_M = _isqrt(M_val)
        valid_ks = [k for k in ks if k <= sqrt_M]
        
        if len(valid_ks) <= best_count:
            continue
        
        # Try to extend: find divisors k > N of M_val such that sigma(k) = M_val // k
        # and k <= sqrt(M_val)
        max_k = sqrt_M
        target = M_val
        
        # Get all divisors of target
        divisors = set()
        i = 1
        while i * i <= target:
            if target % i == 0:
                divisors.add(i)
                divisors.add(target // i)
            i += 1
        
        extended_ks = list(valid_ks)
        existing_ks = set(valid_ks)
        
        for k in sorted(divisors):
            if k > N and k <= max_k and k not in existing_ks:
                expected_sigma = target // k
                if _sigma_by_factorization(k) == expected_sigma:
                    extended_ks.append(k)
                    existing_ks.add(k)
            
            if time.time() > deadline - 25:
                break
        
        if len(extended_ks) > best_count:
            best_M = M_val
            best_ks = sorted(extended_ks)
            best_count = len(extended_ks)
    
    # Final safety: ensure we return valid types
    best_M = int(best_M)
    best_ks = [int(k) for k in best_ks]
    
    return {'M': best_M, 'k': best_ks}

# EVOLVE-BLOCK-END
