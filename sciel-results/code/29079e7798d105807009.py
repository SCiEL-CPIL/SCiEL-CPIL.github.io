import numpy as np
import random

N = 5000

def _primes(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

def _greedy_remove_numpy(S_array, primes_order, rng):
    S_list = S_array
    
    for p in primes_order:
        if p > len(S_list):
            continue
            
        # Count residues mod p using numpy
        residues = S_list % p
        counts = np.bincount(residues, minlength=p)
        
        min_count = np.min(counts)
        
        if min_count == 0:
            continue
            
        # Find all residues with minimum count
        min_indices = np.where(counts == min_count)[0]
        # Pick one randomly to escape deterministic local optima
        r = min_indices[rng.randint(0, len(min_indices) - 1)]
        
        # Remove all elements congruent to r mod p
        mask = residues != r
        S_list = S_list[mask]
        
        if len(S_list) == 0:
            break
            
    return S_list

def construct():
    primes = _primes(N)
    
    best_size = 0
    best_S = np.array([], dtype=int)
    
    # Use a fixed seed for reproducibility in evaluation, 
    # but randomize the prime order for each trial
    rng = random.Random(42)
    
    # Initial array of all integers in [0, N]
    S_init = np.arange(N + 1, dtype=int)
    
    # Run multiple trials with different prime orderings
    # 500 trials should be fast enough with NumPy acceleration
    for _ in range(500):
        p_order = primes[:]
        rng.shuffle(p_order)
        
        S_res = _greedy_remove_numpy(S_init, p_order, rng)
        
        if len(S_res) > best_size:
            best_size = len(S_res)
            best_S = S_res
            
    return best_S.tolist()
