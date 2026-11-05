import numpy as np
import time
import random

def construct():
    # Generate primes up to 1000
    def get_primes(n):
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = [False] * len(sieve[i*i:n+1:i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]
    
    primes = get_primes(1000)
    n_primes = len(primes)
    
    # Evaluation range
    # We aim for L >= 1000. Search up to 2000.
    N = 2000
    
    # Initialize residues: Random
    residues = np.array([random.randint(0, p - 1) for p in primes])
    
    # Initialize counts array
    # counts[x] = number of primes p such that x == r_p (mod p)
    counts = np.zeros(N + 1, dtype=np.int32)
    
    # Populate initial counts using vectorized slicing
    for i, p in enumerate(primes):
        r = residues[i]
        start = r if r > 0 else p
        if start <= N:
            counts[start:N+1:p] += 1
            
    # Objective: Maximize sum of min(counts[x], 2) for x=1..N
    # This penalizes gaps (0 or 1) and rewards full coverage (2+).
    
    def get_score():
        return np.sum(np.minimum(counts[1:], 2))
        
    def get_L():
        bad = np.where(counts[1:] < 2)[0]
        if len(bad) == 0:
            return N
        return bad[0] + 1
        
    current_score = get_score()
    current_L = get_L()
    
    best_score = current_score
    best_L = current_L
    best_residues = residues.copy()
    
    # Simulated Annealing Parameters
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    T = 1.0
    alpha = 0.9999  # Cooling rate
    min_T = 1e-5
    
    iter_count = 0
    
    while time.time() < deadline:
        iter_count += 1
        
        # Pick a random prime index
        i = random.randint(0, n_primes - 1)
        p = primes[i]
        old_r = residues[i]
        
        # Pick a new random residue
        new_r = random.randint(0, p - 1)
        
        if new_r == old_r:
            continue
            
        # Calculate delta in score using vectorized operations
        old_start = old_r if old_r > 0 else p
        new_start = new_r if new_r > 0 else p
        
        # Get indices
        old_indices = np.arange(old_start, N + 1, p)
        new_indices = np.arange(new_start, N + 1, p)
        
        # Get current counts at these indices
        old_counts = counts[old_indices]
        new_counts = counts[new_indices]
        
        # Calculate delta
        # Removal: min(c-1, 2) - min(c, 2)
        delta_rem = np.sum(np.minimum(old_counts - 1, 2) - np.minimum(old_counts, 2))
        # Addition: min(c+1, 2) - min(c, 2)
        delta_add = np.sum(np.minimum(new_counts + 1, 2) - np.minimum(new_counts, 2))
        
        delta = delta_rem + delta_add
        
        # Acceptance criterion
        accept = False
        if delta > 0:
            accept = True
        else:
            # delta is negative
            # Avoid overflow in exp
            if delta / T > -100: # exp(-100) is ~0
                prob = np.exp(delta / T)
                if random.random() < prob:
                    accept = True
                    
        if accept:
            # Update counts using vectorized slicing
            # Note: We must handle the case where old_r and new_r overlap?
            # No, we are changing ONE prime's residue.
            # So we remove old contribution and add new.
            
            # Subtract old
            if len(old_indices) > 0:
                counts[old_indices] -= 1
            # Add new
            if len(new_indices) > 0:
                counts[new_indices] += 1
                
            # Update residues and score
            residues[i] = new_r
            current_score += delta
            
            # Periodically check L
            if iter_count % 50 == 0:
                current_L = get_L()
                if current_L > best_L:
                    best_L = current_L
                    best_residues = residues.copy()
                elif current_score > best_score:
                    best_score = current_score
                    best_residues = residues.copy()
        else:
            # Do nothing
            pass
            
        # Cool down
        T *= alpha
        if T < min_T:
            T = min_T
            
    # Final check
    current_L = get_L()
    if current_L > best_L:
        best_residues = residues.copy()
        
    return list(best_residues)
