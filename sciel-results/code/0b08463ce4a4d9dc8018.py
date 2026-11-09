import time
import numpy as np

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
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # We want to find the largest T.
    # Start with m = N + 1 (size of full set)
    # Actually, max size is N+1.
    m = N + 1
    
    best_points = None
    best_size = 0
    
    while time.time() < deadline:
        R = np.zeros(N + 1, dtype=bool)
        
        # Process primes p <= m
        # Sort primes by ascending order for greedy
        active_primes = [p for p in PRIMES if p <= m]
        
        # Track which residue class was chosen for each prime
        chosen_residues = {}
        
        for p in active_primes:
            # Find r in [0, p-1] minimizing marginal cost
            min_cost = N + 1
            best_r = 0
            
            # For each r, cost = count of x in r, r+p, ... not in R
            # Use numpy for speed
            for r in range(p):
                indices = np.arange(r, N + 1, p)
                cost = np.sum(~R[indices])
                if cost < min_cost:
                    min_cost = cost
                    best_r = r
                    if cost == 0:
                        break # Can't get better than 0
            
            # Add elements of class best_r mod p to R
            indices = np.arange(best_r, N + 1, p)
            R[indices] = True
            chosen_residues[p] = best_r
        
        # Count remaining elements
        remaining = np.sum(~R)
        new_m = int(remaining)
        
        if new_m > best_size:
            best_size = new_m
            best_points = np.where(~R)[0].tolist()
        
        if new_m == m:
            # Converged - no more removals needed
            break
        
        m = new_m
    
    if best_points is not None:
        return best_points
    
    # Fallback: return empty or minimal valid set
    return [0]
