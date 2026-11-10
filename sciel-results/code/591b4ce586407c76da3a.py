import time
import random
import numpy as np

N = 5000
DEADLINE = 320  # safety margin before 330s kill

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

ALL_PRIMES = _primes(N)
ALL_INDICES = np.arange(N + 1)

def _is_admissible(mask, all_primes):
    """Check if the set defined by mask is admissible."""
    size = np.sum(mask)
    if size == 0:
        return True
    
    # Only check primes p <= size
    for p in all_primes:
        if p > size:
            break
        
        # Get indices of active elements
        active_indices = np.where(mask)[0]
        residues = active_indices % p
        counts = np.bincount(residues, minlength=p)
        
        if np.all(counts > 0):
            return False
    return True

def _greedy_remove(mask, all_primes, deadline, random_state=0):
    """
    Greedy removal algorithm with randomization.
    Returns the final mask.
    """
    rng = np.random.RandomState(random_state)
    current_mask = mask.copy()
    current_size = np.sum(current_mask)
    
    while time.time() < deadline:
        if current_size == 0:
            break
            
        # Check admissibility and find bad primes
        bad_primes = []
        for p in all_primes:
            if p > current_size:
                break
            
            active_indices = np.where(current_mask)[0]
            residues = active_indices % p
            counts = np.bincount(residues, minlength=p)
            
            if np.all(counts > 0):
                # Find the minimum count and its index
                min_count = np.min(counts)
                # Get all indices with min count
                min_indices = np.where(counts == min_count)[0]
                bad_primes.append((p, min_count, min_indices))
        
        if not bad_primes:
            # Admissible
            break
        
        # Select a prime to fix
        # Strategy: Pick the one with the smallest min_count (easiest fix)
        # If tie, pick randomly
        min_min_count = min(b[1] for b in bad_primes)
        candidates = [b for b in bad_primes if b[1] == min_min_count]
        
        # Randomly select from candidates
        selected = rng.choice(candidates)
        p, min_count, min_indices = selected
        
        # Randomly select a residue class to remove from the min_indices
        r = rng.choice(min_indices)
        
        # Remove elements x where x % p == r
        active_indices = np.where(current_mask)[0]
        to_remove = active_indices[active_indices % p == r]
        
        if len(to_remove) == 0:
            # Should not happen if counts > 0
            continue
            
        current_mask[to_remove] = False
        current_size -= len(to_remove)
        
        # Optional: If we are stuck in a loop or taking too long, we could break
        # But greedy is monotonic, so it will terminate.
        
    return current_mask

def construct():
    start_time = time.time()
    
    best_mask = np.zeros(N + 1, dtype=bool)
    best_size = 0
    
    # We will run multiple iterations with different random seeds
    # Each iteration runs the greedy removal until admissible
    # We keep the best (largest) result
    
    # Estimate how long one run takes.
    # Usually very fast for N=5000.
    # Let's run as many as possible.
    
    seed = 0
    while time.time() < start_time + DEADLINE - 5:
        # Initial mask: all True
        initial_mask = np.ones(N + 1, dtype=bool)
        
        # Run greedy removal
        result_mask = _greedy_remove(initial_mask, ALL_PRIMES, time.time() + 2, random_state=seed)
        
        size = np.sum(result_mask)
        if size > best_size:
            best_size = size
            best_mask = result_mask.copy()
            
        seed += 1
        
    # Convert best_mask to list of integers
    best_points = np.where(best_mask)[0].tolist()
    
    # Final verification
    if not _is_admissible(best_mask, ALL_PRIMES):
        # Fallback
        best_points = [1]
        
    return best_points
