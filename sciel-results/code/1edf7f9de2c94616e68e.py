"""Improved program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import math
from collections import defaultdict, Counter

N = 5000
START_TIME = time.time()
DEADLINE = 320.0  # safety margin before 330s kill

# EVOLVE-BLOCK-START

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

# Precompute primes up to N
_PRIMES_N = _primes(N)
# Primes p where p*p > N (these are "large" primes that appear in initial set)
_LARGES = [p for p in _PRIMES_N if p * p > N]

def _is_admissible(points, primes_check):
    """Check if points form an admissible tuple w.r.t. primes_check."""
    point_set = set(points)
    for p in primes_check:
        if p > len(point_set):
            continue
        residues = set(v % p for v in point_set)
        if len(residues) == p:
            return False
    return True

def _violating_primes(points, primes_check):
    """Return list of primes for which all residue classes are covered."""
    point_set = set(points)
    viol = []
    for p in primes_check:
        if p > len(point_set):
            continue
        residues = set(v % p for v in point_set)
        if len(residues) == p:
            viol.append(p)
    return viol

def _remove_smallest_class(points, p):
    """Remove the least-populated residue class mod p from points."""
    classes = defaultdict(list)
    for v in points:
        classes[v % p].append(v)
    if len(classes) < p:
        return points  # already admissible for this p
    smallest = min(classes.values(), key=len)
    smallest_set = set(smallest)
    return [v for v in points if v not in smallest_set]

def _greedy_construct(seed_offset=0):
    """Greedy construction: start with 1 and large primes, then repair."""
    rng = random.Random(seed_offset)
    points = [1] + list(_LARGES)
    # Shuffle large primes slightly for variety
    # Actually, keep them sorted for determinism but maybe vary the starting point
    # Let's try different starting points
    # Start point candidates: 0, 1, 2, ... or random small number
    # But 1 is natural. Let's try a few variants.
    
    # Repair loop
    max_iter = 100
    for _ in range(max_iter):
        viol = _violating_primes(points, _PRIMES_N)
        if not viol:
            break
        # Pick the smallest violating prime (hardest to satisfy?)
        # Or pick the one with the smallest minority class to remove?
        # Let's pick the smallest prime first
        p = min(viol)
        points = _remove_smallest_class(points, p)
    
    return points

def _local_search_add(points, max_steps=500):
    """Try to add back removed elements that don't violate admissibility."""
    point_set = set(points)
    # Candidate elements to add: 0..N
    # We want to add elements that are "safe"
    # An element x is safe if for every prime p <= len(points)+1, 
    # adding x doesn't complete all residue classes mod p.
    
    # Actually, adding x is safe if for every p, the current residues mod p 
    # do not already have p-1 distinct residues with x%p being the missing one.
    
    # Let's precompute for each p the set of residues present
    # and which residues are missing
    
    primes_check = [p for p in _PRIMES_N if p <= len(points) + 1]
    
    # For efficiency, maintain residue counts
    # But for simplicity in this step, we can just check
    
    added = 0
    # Try adding from 0 to N, skipping those already in points
    # Randomize order for variety? Or sequential?
    # Let's try sequential first, then random
    
    # Build a list of candidates
    candidates = [i for i in range(N + 1) if i not in point_set]
    
    # For each candidate, check if it's safe
    # To make this faster, we can precompute for each p the missing residues
    # missing_residues[p] = set of residues not in point_set mod p
    
    missing = {}
    for p in primes_check:
        present = set(v % p for v in point_set)
        missing[p] = set(range(p)) - present
    
    safe_candidates = []
    for x in candidates:
        is_safe = True
        for p in primes_check:
            if x % p in missing[p]:
                # Adding x would fill this missing residue
                # Check if this would complete all residues
                # i.e., if missing[p] == {x % p}
                if len(missing[p]) == 1:
                    is_safe = False
                    break
        if is_safe:
            safe_candidates.append(x)
    
    # Add safe candidates
    for x in safe_candidates:
        # Re-check after each addition? That's expensive.
        # Let's add all at once and then verify
        point_set.add(x)
        added += 1
    
    if added == 0:
        return list(point_set)
    
    # After adding, we might have created new violations
    # Re-run repair
    points_new = list(point_set)
    for _ in range(50):
        viol = _violating_primes(points_new, _PRIMES_N)
        if not viol:
            break
        p = min(viol)
        points_new = _remove_smallest_class(points_new, p)
    
    return points_new

def _greedy_with_add_back(seed_offset=0):
    """Greedy construction followed by local search adding back elements."""
    points = _greedy_construct(seed_offset)
    points = _local_search_add(points, max_steps=500)
    return points

def construct():
    """Main construct function with time-bounded search."""
    global START_TIME
    if time.time() - START_TIME > DEADLINE:
        # Fallback to a quick construction
        return _greedy_construct(0)
    
    best_points = _greedy_construct(0)
    best_len = len(best_points)
    
    # Try multiple seeds with local search
    max_seeds = 50
    for seed in range(1, max_seeds + 1):
        if time.time() - START_TIME > DEADLINE - 10:
            break
        
        points = _greedy_with_add_back(seed)
        if len(points) > best_len:
            best_len = len(points)
            best_points = points
    
    # Final verification
    # Ensure the result is valid
    primes_check = [p for p in _PRIMES_N if p <= len(best_points)]
    for p in primes_check:
        residues = set(v % p for v in best_points)
        if len(residues) == p:
            # Fix it
            best_points = _remove_smallest_class(best_points, p)
            break
    
    return best_points

# EVOLVE-BLOCK-END
