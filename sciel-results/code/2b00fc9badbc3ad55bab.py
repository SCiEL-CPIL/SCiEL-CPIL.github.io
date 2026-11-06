import time
import math
import random
import numpy as np

N = 5000

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

# We need primes up to roughly 500-600. 
# Max size is ~450. So primes up to 500 are sufficient.
# Let's precompute up to 1000 to be safe, though only ~95 primes will be used.
MAX_PRIME_FOR_PRECOMP = 1000
ALL_PRIMES = _primes(MAX_PRIME_FOR_PRECOMP)

# Precompute indices for each prime and residue
# prime_indices[p][r] = list of x in [0, N] such that x % p == r
prime_indices = {}
for p in ALL_PRIMES:
    indices = [[] for _ in range(p)]
    for x in range(N + 1):
        r = x % p
        indices[r].append(x)
    prime_indices[p] = indices

def construct():
    start_time = time.time()
    deadline = start_time + 290.0
    
    best_size = 0
    best_set = []
    
    # Use a list of primes sorted by value
    active_primes_list = ALL_PRIMES
    
    # Run multiple greedy attempts
    while time.time() < deadline:
        # Initialize S as all True
        S_active = np.ones(N + 1, dtype=bool)
        current_size = N + 1
        
        # Iterate through primes
        for p in active_primes_list:
            if p > current_size:
                break
            
            # Calculate counts for each residue
            counts = [0] * p
            # Optimization: We can sum S_active over the indices
            # But simple loop is fine for N=5000
            for r in range(p):
                # Count how many active elements are in this residue class
                # Using numpy sum on boolean array slice would be faster if we had a 2D array, 
                # but we have lists.
                # Let's just loop.
                cnt = 0
                for x in prime_indices[p][r]:
                    if S_active[x]:
                        cnt += 1
                counts[r] = cnt
            
            min_count = min(counts)
            
            # If all residues are present, we must remove one
            if min_count > 0:
                # Find all residues with min count
                candidates = [r for r in range(p) if counts[r] == min_count]
                
                # Randomly select one to remove
                # Using random.choice for tie-breaking
                r_to_remove = random.choice(candidates)
                
                # Remove all elements in this residue class
                for x in prime_indices[p][r_to_remove]:
                    if S_active[x]:
                        S_active[x] = False
                        current_size -= 1
        
        # Update best
        if current_size > best_size:
            best_size = current_size
            best_set = [int(x) for x in range(N + 1) if S_active[x]]
            
    return sorted(best_set)
