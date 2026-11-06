"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def can_add(x, S):
    """
    Check if x can be added to set S without violating the condition.
    Condition:
    1. x does not divide the sum of any 3 distinct elements in S.
    2. No element a in S divides the sum of x and any 2 distinct elements in S\{a}.
    """
    # 1. Check if x divides any sum of 3 elements in S
    if len(S) >= 3:
        # Generate sums of 3 distinct elements
        # Using itertools is clear and fast enough for small S
        for b, c, d in combinations(S, 3):
            if (b + c + d) % x == 0:
                return False
    
    # 2. Check if any a in S divides x + b + c for distinct b, c in S\{a}
    for a in S:
        # Get elements in S excluding a
        others = [v for v in S if v != a]
        if len(others) < 2:
            continue
        
        # Check all pairs in others
        for i in range(len(others)):
            bi = others[i]
            for j in range(i + 1, len(others)):
                bj = others[j]
                if (x + bi + bj) % a == 0:
                    return False
    
    return True

def greedy_select(candidates, deadline=None):
    """
    Greedy selection from candidates.
    """
    S = []
    for x in candidates:
        if deadline and time.time() > deadline:
            break
        if can_add(x, S):
            S.append(x)
    return S

def construct():
    start_time = time.time()
    # Safety margin: 55s out of 60s (or 90s if that's the hard kill, but prompt says 60s for eval, 90s kill. Use 50s to be safe).
    deadline = start_time + 50.0
    
    # 1. Initial seed: Greedy from top-down (largest to smallest)
    # This is known to be a strong baseline for this type of problem.
    candidates_seed = list(range(192, 0, -1))
    best_set = greedy_select(candidates_seed, deadline)
    best_size = len(best_set)
    
    if best_size >= 48:
        return best_set
    
    # 2. Randomized restarts
    # We try different orderings to escape local optima.
    # We bias the randomness slightly towards larger numbers in some runs,
    # and use pure random in others.
    
    while time.time() < deadline:
        # Create random candidates
        candidates = list(range(1, 193))
        
        # Strategy: Alternate between pure random and biased random
        # Bias: Sort by random.random() / value? No, just shuffle.
        random.shuffle(candidates)
        
        S = greedy_select(candidates, deadline)
        
        if len(S) > best_size:
            best_set = S
            best_size = len(S)
            if best_size >= 48:
                break
        
        # Optional: Local improvement?
        # Try to swap elements? 
        # Given time constraints, just restarts.
        
    return best_set
# EVOLVE-BLOCK-END
