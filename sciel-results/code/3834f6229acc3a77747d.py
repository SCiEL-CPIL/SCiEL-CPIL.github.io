"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import random
import time

def _check_add(sums, x):
    new_sums = set()
    for s in sums:
        ns = s + Fraction(1, x)
        if ns in sums or ns in new_sums:
            return None
        new_sums.add(ns)
    return new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Strategy 1: Descending Greedy
    candidates = list(range(22, 0, -1))
    current_set = []
    sums = {Fraction(0)}
    for x in candidates:
        new_sums = _check_add(sums, x)
        if new_sums is not None:
            current_set.append(x)
            sums.update(new_sums)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set
        
    # Strategy 2: Random Restart Greedy with Descending Bias
    for seed in range(5000):
        if time.time() > deadline:
            break
        random.seed(seed)
        candidates = list(range(1, 23))
        # Create a biased order: larger numbers are more likely to come first
        # by adding random noise to their value and sorting descending
        biased_candidates = [(x, random.uniform(0, 10)) for x in candidates]
        biased_candidates.sort(key=lambda item: item[0] + item[1], reverse=True)
        candidates = [x for x, _ in biased_candidates]
        
        current_set = []
        sums = {Fraction(0)}
        for x in candidates:
            new_sums = _check_add(sums, x)
            if new_sums is not None:
                current_set.append(x)
                sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            
    return sorted(best_set)
# EVOLVE-BLOCK-END
