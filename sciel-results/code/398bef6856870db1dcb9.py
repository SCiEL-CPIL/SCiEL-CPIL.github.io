"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from math import isqrt

N = 5000

# EVOLVE-BLOCK-START
def _primes(limit):
    if limit < 2:
        return []
    sieve = np.ones(limit + 1, dtype=bool)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            sieve[p*p:limit+1:p] = False
    return np.where(sieve)[0]

def _construct_with_threshold(primes, threshold, use_least_populated):
    """Try a specific sieve threshold and residue strategy. Returns the resulting set size."""
    S_mask = np.ones(N + 1, dtype=bool)
    
    # Sieve small primes up to threshold
    small_primes = primes[primes <= threshold]
    for p in small_primes:
        if use_least_populated:
            # Find least populated residue mod p in current S
            S_indices = np.where(S_mask)[0]
            residues = S_indices % p
            counts = np.bincount(residues, minlength=p)
            best_r = int(np.argmin(counts))
            to_remove = S_indices[residues == best_r]
            S_mask[to_remove] = False
        else:
            # Remove residue 0
            multiples = np.arange(0, N + 1, p)
            S_mask[multiples] = False
    
    remaining_primes = primes[primes > threshold]
    
    changed = True
    while changed:
        S_indices = np.where(S_mask)[0]
        if len(S_indices) == 0:
            break
        
        candidates = remaining_primes[remaining_primes <= len(S_indices)]
        
        bad_p = None
        min_removal_count = float('inf')
        best_r = -1
        best_p_val = -1
        
        for p in candidates:
            residues = S_indices % p
            unique_res = np.unique(residues)
            if len(unique_res) == p:
                counts = np.bincount(residues, minlength=p)
                min_count = int(np.min(counts))
                if min_count < min_removal_count:
                    min_removal_count = min_count
                    best_r = int(np.argmin(counts))
                    best_p_val = p
                    bad_p = p
        
        if bad_p is not None:
            to_remove_mask = (S_indices % best_p_val) == best_r
            values_to_remove = S_indices[to_remove_mask]
            S_mask[values_to_remove] = False
            changed = True
        else:
            changed = False
    
    return np.where(S_mask)[0].tolist()

def construct():
    start_time = time.time()
    deadline = 295.0  # Safety margin
    
    primes = _primes(N)
    
    # Try multiple thresholds and strategies
    best_result = None
    best_size = 0
    
    thresholds = [50, 75, 100, 125, 150, 200]
    strategies = [False, True]  # False = residue 0, True = least populated
    
    for threshold in thresholds:
        for use_least in strategies:
            if time.time() - start_time > deadline:
                break
            result = _construct_with_threshold(primes, threshold, use_least)
            size = len(result)
            if size > best_size:
                best_size = size
                best_result = result
    
    if best_result is None:
        # Fallback: just return all numbers (shouldn't happen)
        best_result = list(range(N + 1))
    
    return best_result

# EVOLVE-BLOCK-END
