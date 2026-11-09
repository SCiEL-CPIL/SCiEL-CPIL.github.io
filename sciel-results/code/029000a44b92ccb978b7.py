"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
import numpy as np

N = 5000
_DEADLINE = time.time() + 320.0  # 330s wall-clock, 10s safety margin

# EVOLVE-BLOCK-START

def _primes(limit):
    """Return list of primes <= limit using Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            for j in range(i * i, limit + 1, i):
                sieve[j] = False
    return [i for i, ok in enumerate(sieve) if ok]

def construct():
    """
    Constructs the largest admissible tuple in [0, N].
    
    Strategy: Backtracking with forward checking.
    
    The problem: find residues r_p for each prime p <= k such that the number of
    integers in [0, N] not congruent to r_p mod p for any p <= k is at least k.
    
    We use backtracking search over the residue assignments, with pruning:
    - Process primes in a specific order (largest first for more flexibility early).
    - At each step, for the current prime, try each residue class.
    - After assigning a residue, compute the current survivor set.
    - Prune if the maximum possible survivors (even with optimal future choices) 
      is less than the target k.
    
    To estimate the maximum possible survivors after partial assignment, we use a 
    lower bound: the current survivor count times the product of (1 - 1/p) for 
    remaining primes. If this lower bound is < k, prune.
    
    We binary search on k, and for each k, run the backtracking to find a valid 
    residue assignment.
    """
    all_primes = _primes(N)
    prime_set = set(all_primes)
    indices = np.arange(N + 1)
    
    # Precompute for each prime the residue masks for speed
    # residue_masks[p] = list of boolean arrays, one per residue
    # But this is too much memory. Instead, compute on the fly with numpy.
    
    def max_possible_survivors(alive, remaining_primes):
        """
        Lower bound on max survivors if we optimally assign residues for remaining_primes.
        Uses the product formula: current_count * prod(1 - 1/p) for p in remaining_primes.
        This is a lower bound because we can always do at least as well as the expected value.
        Actually, it's an approximation. For pruning, we use: if current_count * prod(1-1/p) < k, prune.
        """
        current_count = np.sum(alive)
        if current_count == 0:
            return 0
        # Product of (1 - 1/p) for remaining primes
        prod = 1.0
        for p in remaining_primes:
            prod *= (1.0 - 1.0 / p)
        return int(current_count * prod)
    
    def backtrack(primes_list, target_k, alive, prime_idx, assignments):
        """
        Backtracking search for residue assignments.
        primes_list: list of primes to assign (in order of processing)
        target_k: target size
        alive: current survivor boolean array
        prime_idx: index into primes_list
        assignments: dict mapping prime -> chosen residue
        
        Returns True if a valid assignment is found, updating assignments.
        """
        if time.time() > _DEADLINE - 5:
            return False  # Timeout safety
        
        if prime_idx == len(primes_list):
            # All primes assigned, check if survivors >= target_k
            return np.sum(alive) >= target_k
        
        p = primes_list[prime_idx]
        residues = indices % p
        remaining_primes = primes_list[prime_idx + 1:]
        
        # Try each residue class
        # Sort residues by how many survivors they would remove (ascending: remove fewest first)
        counts = np.bincount(residues, weights=alive, minlength=p)
        # Try residues in order of increasing removal (i.e., decreasing survivors)
        residue_order = np.argsort(counts)  # ascending count = remove fewest first
        
        for r in residue_order:
            if time.time() > _DEADLINE - 5:
                return False
            
            # Check forward: if we choose residue r for prime p, what's the new alive?
            mask = (residues == r)
            new_alive = alive.copy()
            new_alive[mask] = False
            new_count = np.sum(new_alive)
            
            # Pruning: if new_count * prod(1-1/p) for remaining < target_k, skip
            if remaining_primes:
                prod = 1.0
                for pp in remaining_primes:
                    prod *= (1.0 - 1.0 / pp)
                if new_count * prod < target_k:
                    continue
            
            assignments[p] = r
            if backtrack(primes_list, target_k, new_alive, prime_idx + 1, assignments):
                return True
            del assignments[p]
        
        return False
    
    # Binary search on target k
    low = 0
    high = N + 1
    best_k = 0
    best_assignments = None
    
    # We'll try to find the largest k for which a valid assignment exists.
    # For each k, the primes to consider are those <= k.
    # We process them in descending order (largest first) for better pruning.
    
    while low <= high:
        mid = (low + high) // 2
        if time.time() > _DEADLINE - 10:
            break
        
        primes_to_use = [p for p in all_primes if p <= mid]
        # Process in descending order
        primes_to_use.sort(reverse=True)
        
        alive = np.ones(N + 1, dtype=bool)
        assignments = {}
        
        if backtrack(primes_to_use, mid, alive, 0, assignments):
            best_k = mid
            best_assignments = assignments.copy()
            low = mid + 1
        else:
            high = mid - 1
    
    if best_k == 0 or best_assignments is None:
        return []
    
    # Reconstruct the survivor set using the best assignments
    alive = np.ones(N + 1, dtype=bool)
    for p, r in best_assignments.items():
        residues = indices % p
        mask = (residues == r)
        alive[mask] = False
    
    result_indices = np.where(alive)[0]
    
    if len(result_indices) < best_k:
        # Shouldn't happen, but safety
        return result_indices.tolist()
    
    final_set = result_indices[:best_k].tolist()
    return final_set

# EVOLVE-BLOCK-END
