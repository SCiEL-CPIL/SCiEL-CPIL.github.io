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
    TIME_LIMIT = 290  # Safety margin before 300s evaluator kill
    
    # Start with all integers in [0, N]
    alive = np.ones(N + 1, dtype=bool)
    alive_count = N + 1
    
    # Precompute primes up to N+1 (max possible length)
    all_primes = _primes(N + 1)
    
    # Process primes in ASCENDING order.
    # For each prime p <= alive_count, if all p residue classes are present,
    # remove the least-populated residue class to satisfy the constraint.
    
    for p in all_primes:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT:
            break
        
        if p > alive_count:
            break
        
        # Get indices of alive elements
        alive_indices = np.where(alive)[0]
        if len(alive_indices) == 0:
            break
        
        # Compute residues mod p for alive elements
        residues = alive_indices % p
        
        # Count occurrences of each residue class
        counts = np.bincount(residues, minlength=p)
        
        # Check if any residue class is already missed
        if np.min(counts) == 0:
            # Constraint already satisfied for this prime, move on
            continue
        
        # All residue classes are present - need to remove one
        # Remove the least-populated residue class
        min_res = int(np.argmin(counts))
        
        # Identify which alive elements to remove
        to_remove_mask = (residues == min_res)
        remove_indices = alive_indices[to_remove_mask]
        
        # Remove them
        alive[remove_indices] = False
        alive_count -= len(remove_indices)
    
    # Extract final list
    result_indices = np.where(alive)[0]
    result = result_indices.tolist()
    
    # Verification and repair: ensure admissibility
    # Check for any remaining violations and fix them
    n = len(result)
    if n == 0:
        return result
    
    result_set = set(result)
    primes_to_check = _primes(n)
    
    for p in primes_to_check:
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        # Check if all p residues are covered
        residues_covered = set(x % p for x in result_set)
        if len(residues_covered) == p:
            # Need to remove elements from one residue class
            # Find the residue class with fewest elements
            class_members = {}
            for x in result_set:
                r = x % p
                if r not in class_members:
                    class_members[r] = []
                class_members[r].append(x)
            
            # Remove the smallest class
            min_class = min(class_members.values(), key=len)
            for x in min_class:
                result_set.discard(x)
            n = len(result_set)
    
    return sorted(result_set)

# EVOLVE-BLOCK-END
