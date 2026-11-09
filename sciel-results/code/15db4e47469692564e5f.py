"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START

def _sieve_sigma(n):
    """Compute sigma(i) for all i in [1, n] using a sieve."""
    sigma = [0] * (n + 1)
    for i in range(1, n + 1):
        for j in range(i, n + 1, i):
            sigma[j] += i
    return sigma

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill
    
    # Strategy: enumerate k up to K_MAX, compute M = k*sigma(k),
    # find M with the most distinct k solutions where k <= isqrt(M).
    
    # Phase 1: Sieve up to a reasonable bound to find M with many solutions
    # We need k <= isqrt(M) = isqrt(k*sigma(k)), so k <= sigma(k)
    # This is always true for k >= 1 (sigma(k) >= k+1 for k>1, sigma(1)=1)
    # Actually sigma(k) >= k+1 for k > 1, and sigma(1) = 1, so 1*1=1, isqrt(1)=1, ok.
    # For k=1: M=1, k=1 <= isqrt(1)=1. OK.
    # For k>1: sigma(k) > k, so M = k*sigma(k) > k^2, so isqrt(M) >= k. Good.
    
    # Let's try a moderate sieve first
    K1 = 500000  # 500k - should be fast
    sigma = _sieve_sigma(K1)
    
    # Map M -> list of k
    m_to_ks = defaultdict(list)
    for k in range(1, K1 + 1):
        m = k * sigma[k]
        if m <= 10**18:
            m_to_ks[m].append(k)
    
    # Find best M so far
    best_m = 333312
    best_ks = [336, 372, 434]
    best_count = 3
    
    for m, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = m
            best_ks = ks[:]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE * 0.4:
        return {'M': best_m, 'k': best_ks[:]}
    
    # Phase 2: Try a larger sieve
    K2 = 2000000  # 2M
    if time.time() - start_time < DEADLINE * 0.5:
        sigma2 = _sieve_sigma(K2)
        m_to_ks2 = defaultdict(list)
        for k in range(1, K2 + 1):
            m = k * sigma2[k]
            if m <= 10**18:
                m_to_ks2[m].append(k)
        
        for m, ks in m_to_ks2.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks[:]
        
        elapsed = time.time() - start_time
        if elapsed > DEADLINE * 0.7:
            return {'M': best_m, 'k': best_ks[:]}
    
    # Phase 3: Try even larger if time permits
    K3 = 5000000  # 5M
    if time.time() - start_time < DEADLINE * 0.6:
        sigma3 = _sieve_sigma(K3)
        m_to_ks3 = defaultdict(list)
        for k in range(1, K3 + 1):
            m = k * sigma3[k]
            if m <= 10**18:
                m_to_ks3[m].append(k)
        
        for m, ks in m_to_ks3.items():
            if len(ks) > best_count:
                best_count = len(ks)
                best_m = m
                best_ks = ks[:]
    
    # Verify: ensure k_i <= isqrt(M)
    isq = math.isqrt(best_m)
    verified_ks = [k for k in best_ks if k <= isq]
    
    # Also verify k*sigma(k) == M for each
    # For k in our sieve range, we already know sigma(k), but for safety
    # just trust the sieve computation
    
    if not verified_ks:
        return {'M': 333312, 'k': [336, 372, 434]}
    
    return {'M': best_m, 'k': verified_ks[:]}

# EVOLVE-BLOCK-END
