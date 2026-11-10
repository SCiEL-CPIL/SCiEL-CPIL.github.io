import math
import numpy as np
import bisect

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# Precompute primes up to N
PRIMES = _primes(N)

def construct():
    def sieve_size(K):
        idx = bisect.bisect_right(PRIMES, K)
        relevant_primes = PRIMES[:idx]
        
        mask = np.ones(N + 1, dtype=bool)
        
        for p in relevant_primes:
            if not mask.any():
                break
            
            residues = np.arange(N + 1) % p
            counts = np.bincount(residues, weights=mask, minlength=p)
            
            min_count = np.min(counts)
            min_residues = np.where(counts == min_count)[0]
            
            to_remove = np.isin(residues, min_residues)
            mask &= ~to_remove
            
        return np.sum(mask)
        
    def get_sieve_set(K):
        idx = bisect.bisect_right(PRIMES, K)
        relevant_primes = PRIMES[:idx]
        
        mask = np.ones(N + 1, dtype=bool)
        
        for p in relevant_primes:
            if not mask.any():
                break
            
            residues = np.arange(N + 1) % p
            counts = np.bincount(residues, weights=mask, minlength=p)
            
            min_count = np.min(counts)
            min_residues = np.where(counts == min_count)[0]
            
            to_remove = np.isin(residues, min_residues)
            mask &= ~to_remove
            
        return np.where(mask)[0]
        
    # Binary search for the largest K such that sieve_size(K) >= K
    low = 1
    high = 1000
    best_k = 0
    
    while low <= high:
        mid = (low + high) // 2
        size = sieve_size(mid)
        if size >= mid:
            best_k = mid
            low = mid + 1
        else:
            high = mid - 1
            
    if best_k == 0:
        return []
        
    valid_indices = get_sieve_set(best_k)
    S = valid_indices.tolist()
    m = len(S)
    
    # Check if the full set is admissible
    # We know it is for p <= best_k.
    # Check p in (best_k, m]
    
    idx_start = bisect.bisect_right(PRIMES, best_k)
    idx_end = bisect.bisect_right(PRIMES, m)
    
    is_admissible = True
    for p in PRIMES[idx_start:idx_end]:
        residues = set(x % p for x in S)
        if len(residues) == p:
            is_admissible = False
            break
            
    if is_admissible:
        return S
    else:
        return S[:best_k]
