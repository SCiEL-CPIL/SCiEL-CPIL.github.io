import time
import math
import sys
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    max_m = 10**18
    
    # Increase L to find more collisions
    # 5 million is a reasonable upper bound for memory/time in Python with array
    L = 5 * 10**6
    
    # Check if we have enough time for a larger L?
    # 10^6 took ~1-2s for sieve. 5*10^6 should take ~5-10s.
    
    # Sieve for sigma(k)
    # Using array('q') for memory efficiency (8 bytes per element)
    # 5 * 10^6 * 8 = 40 MB
    sigma = array('q', [0]) * (L + 1)
    
    # Compute sum of divisors for all numbers up to L
    # O(L log L)
    for d in range(1, L + 1):
        for m in range(d, L + 1, d):
            sigma[m] += d
            
    # Map M = k * sigma(k) to count and solutions
    counts = {}
    first_k = {}
    sols = {}
    
    for k in range(1, L + 1):
        m_val = k * sigma[k]
        if m_val <= max_m:
            if m_val not in counts:
                counts[m_val] = 1
                first_k[m_val] = k
            else:
                counts[m_val] += 1
                if counts[m_val] == 2:
                    sols[m_val] = [first_k[m_val], k]
                else:
                    sols[m_val].append(k)
                    
    best_m = 0
    best_count = 0
    
    # Find the M with the most solutions
    for m, cnt in counts.items():
        if cnt > best_count:
            best_count = cnt
            best_m = m
            
    # Fallback if no solutions found (should not happen)
    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}
        
    # Get the list of k for the best M
    # If best_count == 1, sols[best_m] might not exist, but we need a list of 1
    # However, the task implies "many solutions", so best_count should be > 1
    # But to be safe:
    if best_m in sols:
        valid_k = sols[best_m]
    else:
        valid_k = [first_k[best_m]]
        
    # Ensure k_i <= sqrt(M)
    sqrt_m = math.isqrt(best_m)
    # Filter just in case, though k <= sigma(k) holds
    valid_k = [k for k in valid_k if k <= sqrt_m]
    valid_k.sort()
    
    M_out = int(best_m)
    k_out = [int(k) for k in valid_k]
    
    return {'M': M_out, 'k': k_out}

if __name__ == "__main__":
    result = construct()
    print(result)
