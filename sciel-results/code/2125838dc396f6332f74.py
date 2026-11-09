import time
import math
import numpy as np

N = 5000

# EVOLVE-BLOCK-START

def _primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]


def construct():
    """Construct the largest admissible tuple in [0, N].
    
    An admissible tuple satisfies: for every prime p <= len(tuple),
    some residue class mod p is completely missed by the tuple.
    """
    start_time = time.time()
    TIME_LIMIT = 280  # Safety margin before 300s evaluator kill
    
    # Start with all integers in [0, N]
    active = np.ones(N + 1, dtype=bool)
    L = N + 1
    
    # Precompute primes up to N+1
    max_len = N + 1
    all_primes = _primes(max_len)
    
    # We process primes in descending order to minimize loss (remove smallest class for largest p)
    # We only need to fix primes p <= L.
    # Since L decreases, we can just iterate through primes in descending order.
    # If p > L, we skip it (it's not constrained).
    
    # To be safe and efficient, we maintain the list of primes to check.
    # Actually, we can just iterate through all_primes in reverse.
    # For each p, if p <= L, we fix it.
    # After fixing, L decreases.
    # We don't need to re-check previous primes (they are satisfied).
    # We don't need to check primes > L (not constrained).
    
    # Optimization: 
    # If we fix p, we remove elements.
    # Do we need to check if other primes are violated?
    # No, as proven, violations don't appear.
    # So we just need to ensure that for all p <= L_final, the condition holds.
    # Since we fix primes in descending order, and L decreases,
    # we might skip some small primes if L drops below them?
    # No, if L drops below p, p is no longer constrained.
    # So we only need to fix p if p <= L at the time we process it.
    
    # Iterate primes in descending order
    for p in reversed(all_primes):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        if p > L:
            continue
        
        # Check if p is violated (all residues present)
        # Get active indices
        active_indices = np.where(active)[0]
        if len(active_indices) == 0:
            break
        
        # Compute residues
        residues = active_indices % p
        # Count occurrences
        counts = np.bincount(residues, minlength=p)
        
        # If any count is 0, it's not violated.
        if np.min(counts) > 0:
            # Find residue with minimum count
            min_res = np.argmin(counts)
            
            # Remove all active x where x % p == min_res
            # We need to update 'active'
            # Find indices to remove
            to_remove = (active_indices % p) == min_res
            remove_indices = active_indices[to_remove]
            
            active[remove_indices] = False
            L -= len(remove_indices)
            
            # If L becomes 0, break
            if L == 0:
                break
    
    # Extract final list
    result_indices = np.where(active)[0]
    return result_indices.tolist()

# EVOLVE-BLOCK-END
