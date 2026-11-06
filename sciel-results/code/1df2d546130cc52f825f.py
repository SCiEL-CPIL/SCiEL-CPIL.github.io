import random
import time
from fractions import Fraction

def _is_dissociated(S):
    """
    Checks if the set S is dissociated.
    """
    if not S:
        return True
    
    k = len(S)
    # Reciprocals as Fractions for exactness
    recips = [Fraction(1, s) for s in S]
    
    # Generate all subset sums
    sums = [Fraction(0)]
    for r in recips:
        sums += [s + r for s in sums]
        
    if len(set(sums)) != len(sums):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin
    
    best_size = 0
    best_set = []
    
    # 1. Deterministic baseline: Greedy from largest to smallest
    # This often finds a good set quickly because large numbers have small reciprocals
    deterministic_pool = list(range(22, 0, -1))
    current_set = []
    for s in deterministic_pool:
        candidate = current_set + [s]
        if _is_dissociated(candidate):
            current_set = candidate
            
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set[:]

    # 2. Randomized search
    pool = list(range(1, 23))
    
    # We will run random restarts until deadline
    # To make it more effective, we can also try to "refine" the best set found
    # by removing one element and trying to add two others, but that's complex.
    # Random greedy is usually sufficient for this size.
    
    restarts = 0
    while time.time() < deadline:
        random.shuffle(pool)
        current_set = []
        
        # Greedy addition
        # We can track how many elements are left to see if we can beat best_size
        remaining = 22
        for s in pool:
            # Pruning: if current size + remaining <= best_size, break
            if len(current_set) + remaining <= best_size:
                break
                
            candidate = current_set + [s]
            if _is_dissociated(candidate):
                current_set = candidate
            remaining -= 1
            
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            # Optional: If we found a new best, we might want to restart with a bias
            # or just continue. Continuing is fine.
            
        restarts += 1
        
    return best_set
