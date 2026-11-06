"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

# Precompute primes up to 1000
def _primes_up_to(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(n**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, n+1, i):
                sieve[j] = False
    return [i for i in range(2, n+1) if sieve[i]]

_ALL_PRIMES = _primes_up_to(1000)
_L_CAP = 100000

def _compute_L(coverage, L_cap=_L_CAP):
    """Compute largest L such that 1..L all covered."""
    # coverage is a numpy bool array of length L_cap
    # Find first False
    false_idx = np.nonzero(~coverage)[0]
    if len(false_idx) == 0:
        return L_cap
    first_false = false_idx[0]
    # L is first_false (since indices 0..L-1 correspond to 1..L)
    return first_false

def _add_class(coverage, p, r, L_cap=_L_CAP):
    """Return new coverage after adding class (p, r)."""
    # Numbers n in 1..L_cap with n % p == r
    # n = r, r+p, r+2p, ... but we need n >= 1
    # If r == 0, n = p, 2p, 3p, ...
    # If r > 0, n = r, r+p, r+2p, ...
    if r == 0:
        start = p
    else:
        start = r
    if start <= L_cap:
        coverage[start-1::p] = True  # 0-indexed: position n-1
    return coverage

def _remove_class(coverage, p, r, L_cap=_L_CAP):
    """Remove class (p, r) from coverage. Note: this is approximate if overlapping."""
    # We can't simply set to False because other classes might cover those numbers.
    # So this function is only valid if we track which numbers are covered by exactly one class,
    # or we rebuild coverage from scratch. For efficiency in local search, we'll rebuild.
    pass

def _rebuild_coverage(classes, L_cap=_L_CAP):
    """Rebuild coverage from list of [p, r] pairs."""
    coverage = np.zeros(L_cap, dtype=bool)
    for p, r in classes:
        if r == 0:
            start = p
        else:
            start = r
        if start <= L_cap:
            coverage[start-1::p] = True
    return coverage

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Initial solution: seed-like
    # Use primes from 13 upward
    primes = [p for p in _ALL_PRIMES if p >= 13]
    classes = [[p, (i + 1) % p] for i, p in enumerate(primes)]
    
    # Compute initial L
    coverage = _rebuild_coverage(classes)
    current_L = _compute_L(coverage)
    
    # Check budget
    budget = sum(Fraction(1, p) for p, r in classes)
    
    # Local search: try to improve L by swapping or adding classes
    # Strategy: 
    # 1. Try to add a new prime class if budget allows
    # 2. Try to swap an existing class with a different residue
    # 3. Try to swap an existing prime with a different prime
    
    best_L = current_L
    best_classes = classes[:]
    
    used_primes = set(p for p, r in classes)
    
    # Precompute for speed
    L_cap = _L_CAP
    
    # Greedy addition: try adding primes not yet used
    # Sort by 1/p (largest p first to save budget? No, smallest p gives most coverage per budget)
    # Actually, for coverage, smaller p covers more numbers. But budget is 1/p.
    # So we want to use small primes.
    
    # Let's try a local search loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try all possible swaps of residue for each existing class
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            p, r = classes[i]
            # Try different residues
            for r_new in range(p):
                if r_new == r:
                    continue
                # Temporarily change
                old_r = r
                classes[i][1] = r_new
                coverage = _rebuild_coverage(classes)
                L_new = _compute_L(coverage)
                if L_new > current_L:
                    current_L = L_new
                    best_L = current_L
                    best_classes = [c[:] for c in classes]
                    improved = True
                else:
                    classes[i][1] = old_r
        
        # Try adding a new prime if budget allows
        if time.time() > deadline:
            break
        for p_new in _ALL_PRIMES:
            if p_new in used_primes:
                continue
            # Check if adding this prime would exceed budget
            # We need to find a residue that helps
            # Try a few residues
            for r_new in [0, 1, p_new - 1, p_new // 2]:
                if time.time() > deadline:
                    break
                # Check budget
                new_budget = budget + Fraction(1, p_new)
                if new_budget > 1:
                    continue
                # Try adding
                classes.append([p_new, r_new])
                coverage = _rebuild_coverage(classes)
                L_new = _compute_L(coverage)
                if L_new > current_L:
                    current_L = L_new
                    best_L = current_L
                    best_classes = [c[:] for c in classes]
                    used_primes.add(p_new)
                    budget = new_budget
                    improved = True
                else:
                    classes.pop()
        
        # Try removing a class and replacing with a different prime
        if time.time() > deadline:
            break
        for i in range(len(classes)):
            if time.time() > deadline:
                break
            p_old, r_old = classes[i]
            # Try replacing with a different prime
            for p_new in _ALL_PRIMES:
                if p_new == p_old or p_new in used_primes:
                    continue
                # Check budget
                # Remove p_old, add p_new
                new_budget = budget - Fraction(1, p_old) + Fraction(1, p_new)
                if new_budget > 1:
                    continue
                # Try a few residues for p_new
                for r_new in [0, 1, p_new - 1, p_new // 2]:
                    if time.time() > deadline:
                        break
                    classes[i] = [p_new, r_new]
                    coverage = _rebuild_coverage(classes)
                    L_new = _compute_L(coverage)
                    if L_new > current_L:
                        current_L = L_new
                        best_L = current_L
                        best_classes = [c[:] for c in classes]
                        used_primes.discard(p_old)
                        used_primes.add(p_new)
                        budget = new_budget
                        improved = True
                    else:
                        classes[i] = [p_old, r_old]
    
    return best_classes
# EVOLVE-BLOCK-END
