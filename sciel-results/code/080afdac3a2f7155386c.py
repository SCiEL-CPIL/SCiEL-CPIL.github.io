"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
from collections import defaultdict

def isqrt(n):
    return math.isqrt(n)

def construct():
    # We need to find M with many k such that k*sigma(k) = M
    # Strategy: enumerate k up to a limit, compute k*sigma(k), group by M
    
    # Sieve for sigma up to a larger limit to find more collisions
    # 10^7 is a good balance between time and coverage
    LIMIT = 10000000
    sigma = [0] * (LIMIT + 1)
    for i in range(1, LIMIT + 1):
        for j in range(i, LIMIT + 1, i):
            sigma[j] += i
    
    # Compute k*sigma(k) for all k and group by M
    m_to_k = defaultdict(list)
    for k in range(1, LIMIT + 1):
        m_val = k * sigma[k]
        if m_val <= 10**18:
            m_to_k[m_val].append(k)
    
    # Find M with most k solutions where k <= isqrt(M)
    best_m = None
    best_k_list = []
    for m_val, k_list in m_to_k.items():
        # Filter k values: must have k <= isqrt(m_val)
        valid_k = [k for k in k_list if k <= isqrt(m_val)]
        if len(valid_k) > len(best_k_list):
            best_k_list = valid_k
            best_m = m_val
    
    if best_m is None:
        # Fallback to known solution
        return {'M': 333312, 'k': [336, 372, 434]}
    
    # Verify all solutions
    verified_k = []
    for k in best_k_list:
        # Compute sigma(k) exactly
        s = 0
        n = k
        temp = n
        d = 2
        while d * d <= temp:
            if temp % d == 0:
                p_pow = 1
                while temp % d == 0:
                    p_pow *= d
                    temp //= d
                s_factor = (p_pow * d - 1) // (d - 1)
                s = 0 if s == 0 else s * s_factor
                if s == 0:
                    s = s_factor
            d += 1
        if temp > 1:
            s_factor = 1 + temp
            s = 0 if s == 0 else s * s_factor
            if s == 0:
                s = s_factor
        if s == 0:
            s = 1
        if k * s == best_m:
            verified_k.append(k)
    
    # Ensure distinct and sorted
    verified_k = sorted(set(verified_k))
    
    if len(verified_k) >= 3:
        return {'M': int(best_m), 'k': [int(k) for k in verified_k]}
    else:
        # Fallback
        return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
