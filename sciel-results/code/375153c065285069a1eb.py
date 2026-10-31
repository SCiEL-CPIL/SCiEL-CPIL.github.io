import numpy as np
import time
import random

N = 50

def _is_sidon_extension(S, k, sums):
    """Check if k can be added to S while maintaining Sidon property."""
    # New sums are s+k for s in S, and k+k
    # Check if any new sum is in existing sums
    for s in S:
        if s + k in sums:
            return False
    if k + k in sums:
        return False
    return True

def _greedy_sidon(N, order):
    S = []
    sums = set()
    for k in order:
        if _is_sidon_extension(S, k, sums):
            S.append(k)
            # Add new sums
            for s in S:
                sums.add(s + k)
            sums.add(k + k)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 110.0 # 10s margin
    
    best_set = []
    
    # Try standard greedy
    best_set = _greedy_sidon(N, list(range(N)))
    
    # Try random restarts
    iterations = 0
    while time.time() < deadline:
        random.seed(iterations) # Or just use global random
        order = list(range(N))
        random.shuffle(order)
        S = _greedy_sidon(N, order)
        if len(S) > len(best_set):
            best_set = S
        
        # Occasionally try a local search or different heuristic?
        # For now, random shuffles are good.
        iterations += 1
        
    # If we found a set, use it. Else fallback to single spike.
    if not best_set:
        return [1.0] + [0.0] * (N - 1)
        
    K = len(best_set)
    h = [0.0] * N
    for i in best_set:
        h[i] = 1.0 / K
        
    return h
