"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 5000

import time
import math
import random

_START_TIME = time.time()
_DEADLINE = 320.0

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

def construct():
    deadline = _START_TIME + _DEADLINE
    primes_cache = {}
    
    def get_primes(limit):
        if limit not in primes_cache:
            primes_cache[limit] = _primes(limit)
        return primes_cache[limit]
    
    # Precompute primes up to N
    all_primes = get_primes(N)
    primes_set = set(all_primes)
    
    # Start with a heuristic set: 0 and all primes > sqrt(N)
    points = set()
    points.add(0)
    for p in all_primes:
        if p * p > N:
            points.add(p)
    
    # Phase 1: Repair initial set
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = False
        current_size = len(points)
        for p in get_primes(current_size):
            residue_counts = {}
            for v in points:
                r = v % p
                residue_counts[r] = residue_counts.get(r, 0) + 1
            if len(residue_counts) == p:
                # All residues present, remove from least populated
                min_count = min(residue_counts.values())
                target = [r for r, c in residue_counts.items() if c == min_count][0]
                to_remove = [v for v in points if v % p == target]
                removed = min(to_remove)
                points.remove(removed)
                changed = True
                break
    
    # Phase 2: Greedy addition with randomized restarts
    best_points = set(points)
    best_size = len(best_points)
    
    # Multiple restarts with different random seeds
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Reset points to initial heuristic
        points = set()
        points.add(0)
        for p in all_primes:
            if p * p > N:
                points.add(p)
        
        # Repair
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            current_size = len(points)
            for p in get_primes(current_size):
                residue_counts = {}
                for v in points:
                    r = v % p
                    residue_counts[r] = residue_counts.get(r, 0) + 1
                if len(residue_counts) == p:
                    min_count = min(residue_counts.values())
                    target = [r for r, c in residue_counts.items() if c == min_count][0]
                    to_remove = [v for v in points if v % p == target]
                    removed = min(to_remove)
                    points.remove(removed)
                    changed = True
                    break
        
        # Greedy addition with randomization
        all_candidates = list(range(0, N + 1))
        
        # Randomize order based on restart
        rng = random.Random(restart * 12345 + 6789)
        rng.shuffle(all_candidates)
        
        for cand in all_candidates:
            if time.time() > deadline:
                break
            if cand in points:
                continue
            # Check if adding cand keeps set admissible
            new_size = len(points) + 1
            valid = True
            for p in get_primes(new_size):
                # Check if cand completes a residue class
                # For efficiency, check if there's already a point with all other residues covered
                # Simpler: just check if after adding, all residues are present
                residues = set()
                for v in points:
                    residues.add(v % p)
                if (len(residues) == p - 1) and ((cand % p) not in residues):
                    valid = False
                    break
            if valid:
                points.add(cand)
        
        # Update best
        if len(points) > best_size:
            best_size = len(points)
            best_points = set(points)
    
    return list(best_points)
