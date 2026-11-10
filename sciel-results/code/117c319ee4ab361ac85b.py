import numpy as np
import time

def _sigma_prime_power(p, a):
    """Compute sigma(p^a) = (p^(a+1) - 1) / (p - 1)"""
    if a == 0:
        return 1
    return (p**(a+1) - 1) // (p - 1)

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    fallback_M = 333312
    fallback_k = [336, 372, 434]
    
    # Strategy: Meet-in-the-middle with smooth numbers
    # k = k_3smooth * l_5smooth
    # k_3smooth = 2^a * 3^b
    # l_5smooth = 5^c
    
    # Limits
    # We want k * sigma(k) * l * sigma(l) <= 10^18
    # Max k, l roughly 10^9
    
    # Generate 3-smooth numbers
    smooth_3 = []
    # 2^a * 3^b <= 10^9
    # a from 0 to 30
    for a in range(31):
        pow2 = 2**a
        if pow2 > 10**9: break
        for b in range(20): # 3^19 > 10^9
            pow3 = 3**b
            if pow2 * pow3 > 10**9: break
            k_val = pow2 * pow3
            # sigma(k_val) = sigma(2^a) * sigma(3^b)
            sig_k = _sigma_prime_power(2, a) * _sigma_prime_power(3, b)
            A_k = k_val * sig_k
            if A_k > 10**18: continue
            smooth_3.append((k_val, A_k))
            
    # Generate 5-smooth numbers (powers of 5)
    smooth_5 = []
    for c in range(20): # 5^13 > 10^9
        pow5 = 5**c
        if pow5 > 10**9: break
        l_val = pow5
        sig_l = _sigma_prime_power(5, c)
        B_l = l_val * sig_l
        if B_l > 10**18: continue
        smooth_5.append((l_val, B_l))
        
    # Count M values
    # M = A_k * B_l
    # We want to find M with most pairs (k, l)
    # Note: k_final = k_val * l_val
    # Distinct k_final?
    # Since gcd(k_val, l_val) = 1 (primes 2,3 vs 5), k_final is unique for each pair (k_val, l_val).
    
    m_counts = {}
    m_pairs = {}
    
    for k_val, A_k in smooth_3:
        for l_val, B_l in smooth_5:
            M = A_k * B_l
            if M > 10**18:
                continue
            
            k_final = k_val * l_val
            # Check constraint k_final <= sqrt(M)
            # As derived, this is always true for this construction.
            # But let's be safe with integer arithmetic
            if k_final * k_final > M:
                continue
                
            if M not in m_counts:
                m_counts[M] = 0
                m_pairs[M] = []
            m_counts[M] += 1
            m_pairs[M].append(k_final)
            
            # Time check
            if time.time() > deadline - 10:
                break
        if time.time() > deadline - 10:
            break
            
    if not m_counts:
        return {'M': fallback_M, 'k': fallback_k}
        
    # Find M with max count
    best_M = max(m_counts, key=m_counts.get)
    best_k_list = m_pairs[best_M]
    
    # Check if we beat the fallback
    if len(best_k_list) > len(fallback_k):
        return {'M': int(best_M), 'k': [int(k) for k in best_k_list]}
    else:
        return {'M': fallback_M, 'k': fallback_k}
