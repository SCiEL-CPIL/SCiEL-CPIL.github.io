import time
import numpy as np
import random
from math import gcd
from functools import reduce

# EVOLVE-BLOCK-START

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_dissociated_subset(S, recips_int=None, lcm_val=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    Uses exact integer arithmetic via LCM scaling.
    
    For subset S, compute LCM of all elements.
    Scale: 1/s -> LCM/s (integer)
    All subset sums are integers -> exact collision detection via set.
    """
    if not S:
        return True
    
    n = len(S)
    
    if recips_int is None or lcm_val is None:
        # Compute LCM of S
        lcm_val = reduce(_lcm, S)
        recips_int = [lcm_val // s for s in S]
    
    # Generate all subset sums using integers
    sums = {0}
    for r in recips_int:
        # Add r to all existing sums
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums or ns in new_sums:
                return False  # Collision detected early
            new_sums.add(ns)
        sums.update(new_sums)
        
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin for 90s limit
    
    # Precompute reciprocals as integers for each possible subset
    # For efficiency, precompute LCMs and scaled values for all subsets
    # of various sizes we might encounter
    
    universe = list(range(1, 23))
    
    # Known good seed (size 9)
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Initialize random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Cache for LCM computations
    lcm_cache = {}
    
    def get_lcm_for_subset(S):
        key = tuple(sorted(S))
        if key in lcm_cache:
            return lcm_cache[key]
        val = reduce(_lcm, key)
        lcm_cache[key] = val
        return val
    
    # Strategy 1: Greedy with various orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add random permutations
    max_greedy_randoms = 80
    for _ in range(max_greedy_randoms):
        if time.time() > deadline - 25:  # Leave 25s for random sampling
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 25:
            break
            
        current_set = []
        used = set()
        
        for x in order:
            if x in used:
                continue
            # Try adding x
            new_set = current_set + [x]
            if _is_dissociated_subset(new_set):
                current_set = new_set
                used.add(x)
                
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we reached a very high number
        if best_size >= 15:
            break

    # Strategy 2: Random sampling of specific sizes
    # Try to find sets of size > best_size by random sampling.
    
    max_target = min(22, max(best_size + 1, 14))
    
    for target_size in range(max_target, best_size, -1):
        if time.time() > deadline - 5:
            break
        
        # Number of attempts
        if target_size >= 14:
            max_attempts = 3000
        elif target_size >= 12:
            max_attempts = 15000
        else:
            max_attempts = 50000
            
        attempts = 0
        found = False
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            
            # Random subset of target_size
            candidate = random.sample(universe, target_size)
            if _is_dissociated_subset(candidate):
                best_set = candidate
                best_size = target_size
                found = True
                break
            attempts += 1
        
        # If we found a valid set in this target_size, break the outer loop
        if found:
            break

    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
