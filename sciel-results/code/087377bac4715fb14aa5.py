"""Rewritten program with a stronger algorithm for covering 1..L by prime residue classes."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from fractions import Fraction

_START_TIME = time.time()
_TIME_LIMIT = 115.0  # Safety margin before 120s/150s kill

def _is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _get_primes(limit=1000):
    return [p for p in range(2, limit + 1) if _is_prime(p)]

def _compute_coverage(primes, residues, cap=100000):
    """
    Returns the largest L such that every integer 1..L is covered.
    Uses numpy for fast marking.
    """
    if not primes:
        return 0
    
    covered = np.zeros(cap + 1, dtype=bool)
    for p, r in zip(primes, residues):
        if r == 0:
            start = p
        else:
            start = r
        if start <= cap:
            covered[start::p] = True
            
    # Find longest prefix of True starting from 1
    # covered[1] is index 1, covered[2] is index 2, etc.
    # We want the largest L such that covered[1:L+1] are all True.
    # If covered[1] is False, L=0.
    # We can use the fact that once we find a False, we stop.
    # Vectorized approach: find first False in covered[1:].
    # If all are True, L = cap.
    # np.where(covered[1:] == False)[0] gives indices of False.
    # If empty, L = cap.
    # Else, first_false_idx is the index in covered[1:], so the actual value is first_false_idx + 1.
    # L = first_false_idx.
    
    false_indices = np.where(covered[1:] == False)[0]
    if len(false_indices) == 0:
        return cap
    else:
        return false_indices[0]

def construct():
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback to a known valid solution
        primes = [p for p in range(13, 1001) if _is_prime(p)]
        return [[p, (i + 1) % p] for i, p in enumerate(primes)]

    all_primes = _get_primes(1000)
    rng = random.Random(42)
    
    # Strategy:
    # 1. Start with a good set of primes. The seed uses primes >= 13.
    #    Sum(1/p) for p=13..997 is approx 0.92.
    #    Let's try to include smaller primes if possible, or just optimize residues for the seed set first.
    #    Actually, including small primes like 2, 3, 5, 7 is very powerful for coverage.
    #    But budget is tight.
    #    Let's stick to the seed's prime set initially for stability, then maybe swap later?
    #    Swapping primes is risky and complex. Let's focus on residue optimization first.
    
    primes = [p for p in range(13, 1001) if _is_prime(p)]
    residues = [(i + 1) % p for i, p in enumerate(primes)]
    
    best_L = _compute_coverage(primes, residues)
    best_primes = primes[:]
    best_residues = residues[:]
    
    # Local search: Simulated Annealing or simple Hill Climbing with random restarts/perturbations
    # Hill climbing: change one residue at a time, accept if improves.
    # To escape local minima, we can occasionally accept worse moves (SA) or do random restarts of residues.
    
    # Let's use a simple stochastic hill climbing with a "temperature" that cools down.
    # Or just a long run of single-residue changes.
    
    current_L = best_L
    cur_primes = primes[:]
    cur_residues = residues[:]
    
    # Precompute coverage array for current state? No, _compute_coverage is fast enough with numpy.
    # But calling it every step might be overhead. 
    # Optimization: Keep a boolean array `covered` and update it incrementally?
    # Changing one residue (p, r_old) to (p, r_new) means:
    # 1. Unmark all n where n%p == r_old (only if they weren't covered by others).
    # 2. Mark all n where n%p == r_new.
    # This is tricky because "unmarking" requires knowing if other primes cover them.
    # So we'd need to recompute coverage for those specific spots or just recompute the whole thing.
    # Given cap=10^5 and ~160 primes, full recompute is ~1.6e7 ops. In numpy, this is very fast (<1ms).
    # So full recompute is fine.
    
    # Let's try a more directed search:
    # Find the first uncovered number U.
    # For each prime p, we can set residue r = U % p to cover U.
    # This is a "greedy repair" step.
    
    # Algorithm:
    # 1. Compute L.
    # 2. If L == cap, return.
    # 3. U = L + 1.
    # 4. For each prime p in the set:
    #    a. Save old residue.
    #    b. Set residue to U % p.
    #    c. Compute new L.
    #    d. If new L > current L, accept.
    #    e. Else, revert.
    # 5. If no improvement found in step 4, try random perturbation of a random residue.
    # 6. Repeat until time limit.
    
    while time.time() - _START_TIME < _TIME_LIMIT:
        L = _compute_coverage(cur_primes, cur_residues)
        if L == 100000:
            break
            
        if L > best_L:
            best_L = L
            best_primes = cur_primes[:]
            best_residues = cur_residues[:]
            
        U = L + 1
        
        improved = False
        # Try to cover U by changing one prime's residue
        # Iterate primes in random order to avoid bias
        indices = list(range(len(cur_primes)))
        rng.shuffle(indices)
        
        for idx in indices:
            p = cur_primes[idx]
            old_r = cur_residues[idx]
            new_r = U % p
            
            if new_r == old_r:
                continue
                
            cur_residues[idx] = new_r
            new_L = _compute_coverage(cur_primes, cur_residues)
            
            if new_L > L:
                L = new_L
                improved = True
                # We found an improvement. We can break or continue to see if we can cover more.
                # Let's break and restart the loop to re-evaluate U.
                break
            else:
                cur_residues[idx] = old_r
        
        if not improved:
            # Local minimum or stuck.
            # Try random perturbation
            idx = rng.randrange(len(cur_primes))
            p = cur_primes[idx]
            old_r = cur_residues[idx]
            new_r = rng.randrange(p)
            cur_residues[idx] = new_r
            new_L = _compute_coverage(cur_primes, cur_residues)
            
            if new_L > L:
                L = new_L
                # Accept
            else:
                # Revert (Hill climbing)
                # Or accept with small probability (Simulated Annealing)?
                # For now, simple hill climbing.
                cur_residues[idx] = old_r
                
            # If we are stuck for a long time, maybe restart residues randomly?
            # Let's add a counter for stuck iterations.
            # For simplicity, just keep going.
            
    return [[p, r] for p, r in zip(best_primes, best_residues)]
# EVOLVE-BLOCK-END
