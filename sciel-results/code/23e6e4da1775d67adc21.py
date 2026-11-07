"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _sigma(n):
    """Compute sum of divisors of n using trial division."""
    if n == 1:
        return 1
    result = 1
    d = 2
    m = n
    while d * d <= m:
        if m % d == 0:
            term = 1
            p_power = 1
            while m % d == 0:
                m //= d
                p_power *= d
                term += p_power
            result *= term
        d += 1 if d == 2 else 2  # 2, 3, 5, 7, ...
    if m > 1:
        result *= (1 + m)
    return result

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    best_score = 0
    best_result = {'M': 333312, 'k': [336, 372, 434]}
    
    # Phase 1: Direct enumeration up to a reasonable limit
    # For k up to K, compute M = k * sigma(k) and bucket
    # We want k <= sqrt(M), so k <= sqrt(k * sigma(k)) => k <= sigma(k), always true
    # Actually k <= sqrt(M) means k^2 <= M = k*sigma(k) => k <= sigma(k), which is always true
    # So any k we find is automatically <= sqrt(M)
    
    # Enumerate k from 1 to K
    K = 2000000  # 2 million, should be fast enough
    buckets = defaultdict(list)
    
    for k in range(1, K + 1):
        if time.time() > deadline:
            break
        s = _sigma(k)
        M = k * s
        if M <= 10**18:
            buckets[M].append(k)
    
    # Find the M with most solutions
    for M, ks in buckets.items():
        if len(ks) > len(best_result['k']):
            best_score = len(ks)
            best_result = {'M': int(M), 'k': [int(k) for k in ks]}
        if time.time() > deadline:
            break
    
    # Phase 2: If we have time, try a larger range with a different approach
    # For highly composite numbers, sigma can be large, so M can be large
    # Try k up to 5 million
    if time.time() < deadline - 20:
        K2 = 5000000
        # Continue from where we left off
        for k in range(K + 1, K2 + 1):
            if time.time() > deadline - 5:
                break
            s = _sigma(k)
            M = k * s
            if M <= 10**18:
                buckets[M].append(k)
        
        # Re-check for best
        for M, ks in buckets.items():
            if len(ks) > len(best_result['k']):
                best_score = len(ks)
                best_result = {'M': int(M), 'k': [int(k) for k in ks]}
    
    # Ensure types are correct
    best_result['M'] = int(best_result['M'])
    best_result['k'] = [int(k) for k in best_result['k']]
    
    return best_result
# EVOLVE-BLOCK-END
