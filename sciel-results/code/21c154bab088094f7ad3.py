"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def _linear_sigma_sieve(n):
    """Compute sigma(k) for all k in [1, n] using a linear sieve in O(n) time."""
    sigma = [0] * (n + 1)
    sigma[1] = 1
    primes = []
    is_prime = [True] * (n + 1)
    # p_power[k] = highest power of smallest prime dividing k
    # p_sigma[k] = sigma of the highest power of smallest prime dividing k
    p_power = [0] * (n + 1)
    p_sigma = [0] * (n + 1)
    
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)
            sigma[i] = 1 + i
            p_power[i] = i
            p_sigma[i] = 1 + i
        for p in primes:
            if i * p > n:
                break
            is_prime[i * p] = False
            if i % p == 0:
                # p divides i, so p is the smallest prime factor of i*p
                p_power[i * p] = p_power[i] * p
                p_sigma[i * p] = p_sigma[i] + p_power[i * p]
                sigma[i * p] = sigma[i] // p_sigma[i] * p_sigma[i * p]
                break
            else:
                # p does not divide i
                p_power[i * p] = p
                p_sigma[i * p] = 1 + p
                sigma[i * p] = sigma[i] * (1 + p)
    return sigma

def _construct_sieve():
    """Sieve-based enumeration of k*sigma(k) to find M with many solutions."""
    deadline = time.time() + 130  # safety margin before 150s kill
    
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3
    
    for max_k in [5_000_000, 10_000_000, 20_000_000, 50_000_000, 100_000_000]:
        if time.time() > deadline - 20:
            break
        
        # Compute sigma using linear sieve (O(n) time)
        sigma = _linear_sigma_sieve(max_k)
        
        if time.time() > deadline - 10:
            break
        
        # Now compute M = k * sigma(k) and find collisions
        M_to_ks = defaultdict(list)
        for k in range(1, max_k + 1):
            M = k * sigma[k]
            if M > 10**18:
                continue
            # Check k <= isqrt(M)
            if k * k <= M:  # k <= sqrt(M)
                M_to_ks[M].append(k)
        
        if time.time() > deadline - 5:
            break
        
        # Find the M with the most k's
        for M, ks in M_to_ks.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_M = M
                best_k = ks[:1000]
        
        if best_count >= 10:
            break  # Good enough for now
    
    if best_count > 3:
        return {'M': best_M, 'k': best_k}
    else:
        return {'M': 333312, 'k': [336, 372, 434]}

def construct():
    deadline = time.time() + 140
    try:
        result = _construct_sieve()
    except Exception:
        result = {'M': 333312, 'k': [336, 372, 434]}
    return result
# EVOLVE-BLOCK-END
