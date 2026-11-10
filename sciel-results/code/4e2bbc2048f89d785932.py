"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = [False] * len(sieve[i*i::i])
    return np.array([i for i, is_p in enumerate(sieve) if is_p])

def _score(residues, primes, limit=100000):
    """
    Compute the largest L such that every integer 1..L satisfies at least two congruences.
    residues: array of 168 residues
    primes: array of 168 primes
    """
    # We need to check coverage for integers 1..limit
    # An integer x is covered by prime index i if x % primes[i] == residues[i]
    # We want the count of covering primes for each x in 1..limit
    # Then find the longest prefix where min count >= 2
    
    # Vectorized approach:
    # For each prime i, the residues x in [1, limit] covered are:
    # r_i, r_i + p_i, r_i + 2*p_i, ... <= limit
    # We can use a difference array or direct accumulation.
    
    # Efficient method: use numpy to create an array of counts
    # counts[x] = number of primes p_i such that x % p_i == r_i
    
    # Direct vectorization might be memory intensive for limit=100000 and 168 primes.
    # 168 * 100000 = 16.8 million operations, which is fine for numpy.
    
    # Let's build the counts array
    # x = 1..limit
    # For each prime i, mark x where x % p_i == r_i
    
    # Alternative: iterate over primes and add to counts
    # This is O(N * limit / p_i) total, which is manageable.
    
    counts = np.zeros(limit + 1, dtype=np.int32)
    
    for i in range(len(primes)):
        p = primes[i]
        r = residues[i]
        # Find all x in [1, limit] such that x % p == r
        # x = r + k*p
        # Smallest x >= 1:
        # If r == 0, x = p, 2p, ...
        # If r > 0, x = r, r+p, ... (but r < p, so r >= 1 unless r=0)
        
        if r == 0:
            start = p
        else:
            start = r
        
        if start <= limit:
            # Generate sequence start, start+p, start+2p, ... <= limit
            # np.arange(start, limit + 1, p)
            idx = np.arange(start, limit + 1, p)
            counts[idx] += 1
    
    # Now find the largest L such that counts[1..L] are all >= 2
    # Find first index where counts < 2
    # We want the longest prefix starting from 1
    # If counts[1] < 2, L = 0
    # Else, find first j >= 1 where counts[j] < 2, then L = j - 1
    
    # Create a boolean mask: valid[j] = counts[j] >= 2
    valid = counts[1:] >= 2  # indices 1..limit
    
    if not np.any(valid):
        return 0
    
    # Find the first False in the prefix
    # If all True, L = limit
    # Else, find first index where valid is False
    # np.where(valid == False)[0] gives indices in valid array (0-based)
    # The first such index k means valid[k] is False, so counts[k+1] < 2
    # Thus L = k (since counts[1..k] are all >= 2)
    
    false_indices = np.where(valid == False)[0]
    if len(false_indices) == 0:
        return limit
    else:
        first_false = false_indices[0]
        return first_false  # L = first_false, because valid[0..first_false-1] are True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    primes = _primes_up_to(1000)
    n_primes = len(primes)
    
    # Initial random assignment
    residues = np.random.randint(0, primes, size=n_primes)
    best_residues = residues.copy()
    best_score = _score(residues, primes)
    
    # Simulated Annealing
    T = 50.0
    T_min = 0.1
    cooling_rate = 0.999
    
    rng = np.random.default_rng(42)
    
    while time.time() < deadline:
        if T < T_min:
            T = T_min
        
        # Make a random move: change 1-3 residues
        n_moves = rng.integers(1, 4)
        indices = rng.choice(n_primes, size=n_moves, replace=False)
        old_values = residues[indices].copy()
        new_values = rng.integers(0, primes[indices], size=n_moves)
        residues[indices] = new_values
        
        new_score = _score(residues, primes)
        
        if new_score >= best_score:
            best_score = new_score
            best_residues = residues.copy()
        else:
            # Accept with probability exp((new_score - best_score) / T)
            delta = new_score - best_score
            if delta < 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    pass  # Accept
                else:
                    # Reject: revert
                    residues[indices] = old_values
        
        T *= cooling_rate
    
    # Return best_residues as a list of integers
    return best_residues.tolist()
# EVOLVE-BLOCK-END
