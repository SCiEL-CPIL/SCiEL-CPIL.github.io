"""Additive base with few representations: randomized local search with smart initialization.

Uses multiple random initializations and aggressive local search to find asymmetric
structures with lower peak multiplicity than symmetric radix designs.
"""
import time
import random
import numpy as np

# Fixed deadline guard (evaluator kills at 90 s wall-clock; keep a large margin).
_DEADLINE = time.time() + 70.0


def _compute_multiplicities(A):
    """Compute representation counts for all s in [128,256]. Returns dict."""
    A_set = set(A)
    mult = {}
    for s in range(128, 257):
        count = 0
        for a in A:
            b = s - a
            if b >= a and b in A_set:
                count += 1
        mult[s] = count
    return mult


def _peak(A):
    """Largest unordered-pair representation multiplicity over [128,256]."""
    mult = _compute_multiplicities(A)
    return max(mult.values()) if mult else 0


def _is_valid(A):
    """True iff every s in [128,256] is a sum of two elements of A."""
    mult = _compute_multiplicities(A)
    return all(v > 0 for v in mult.values())


def _generate_random_valid_set(min_size=30, max_size=80):
    """Generate a random set that is likely to cover [128,256]."""
    # Start with a guaranteed valid base
    base = set(range(64, 129))
    
    # Randomly remove some elements while maintaining validity
    A = set(base)
    while len(A) > min_size and time.time() < _DEADLINE - 5:
        candidates = list(A)
        random.shuffle(candidates)
        removed = False
        for elem in candidates:
            new_A = A - {elem}
            if _is_valid(new_A):
                A = new_A
                removed = True
                break
        if not removed:
            break
    
    # Optionally add some random elements to explore different structures
    if len(A) < max_size:
        for v in range(0, 129):
            if v not in A and random.random() < 0.3:
                A.add(v)
                if len(A) >= max_size:
                    break
    
    return A


def _local_search(A, max_iterations=5000, time_budget=None):
    """Aggressive local search: try removals and additions to minimize peak."""
    if time_budget is None:
        time_budget = _DEADLINE - time.time() - 5
    
    A = list(set(A))
    peak = _peak(A)
    best_A = A[:]
    best_peak = peak
    
    for iteration in range(max_iterations):
        if time.time() > _DEADLINE - 5:
            break
        
        # Try removing a random element
        if len(A) > 10:
            idx = random.randint(0, len(A) - 1)
            cand = A[:idx] + A[idx + 1:]
            if _is_valid(cand):
                p = _peak(cand)
                if p < peak:
                    A = cand
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < 0.1:
                    # Accept equal peak with small probability to escape local optima
                    A = cand
        
        # Try adding a random element
        if len(A) < 128:
            v = random.randint(0, 128)
            if v not in A:
                cand = A + [v]
                p = _peak(cand)
                if p < peak:
                    A = cand
                    peak = p
                    best_A = A[:]
                    best_peak = peak
                elif p == peak and random.random() < 0.05:
                    # Accept equal peak with very small probability
                    A = cand
        
        # Try swapping: remove one, add another
        if len(A) > 5 and len(A) < 127:
            idx = random.randint(0, len(A) - 1)
            v = random.randint(0, 128)
            if v not in A:
                cand = A[:idx] + A[idx + 1:] + [v]
                if _is_valid(cand):
                    p = _peak(cand)
                    if p < peak:
                        A = cand
                        peak = p
                        best_A = A[:]
                        best_peak = peak
                    elif p == peak and random.random() < 0.1:
                        A = cand
    
    return best_A, best_peak


def construct():
    t0 = time.time()
    random.seed(42)  # Reproducibility
    
    best_A = None
    best_peak = float('inf')
    
    # Multiple random restarts
    num_restarts = 10
    for restart in range(num_restarts):
        if time.time() > _DEADLINE - 10:
            break
        
        # Generate a random valid starting set
        A = _generate_random_valid_set(min_size=25, max_size=70)
        
        if not _is_valid(A):
            continue
        
        # Run local search
        A_opt, peak = _local_search(A, max_iterations=3000)
        
        if peak < best_peak:
            best_peak = peak
            best_A = A_opt
        
        # Early exit if we found optimal
        if best_peak <= 1:
            break
    
    # If no valid set found, use fallback
    if best_A is None:
        best_A = list(range(64, 129))
        best_peak = _peak(best_A)
    
    return sorted(best_A)
