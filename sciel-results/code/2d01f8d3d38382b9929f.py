"""Search for maximum-cardinality reciprocal-dissociated subset of [1..22]."""
import time
import sys
import random
from math import gcd
from functools import reduce

# LCM of 1..22
def _lcm(a, b):
    return a * b // gcd(a, b)

_LCM = reduce(_lcm, range(1, 23))
# LCM = 232792560

# Weights: w_i = LCM // i for i in 1..22
_WEIGHTS = [None] + [_LCM // i for i in range(1, 23)]

def _verify(subset):
    """Check if all 2^k subset sums of weights are distinct."""
    k = len(subset)
    sums = {0}
    for i in subset:
        w = _WEIGHTS[i]
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
    return True

def _can_add(sums, w):
    """Check if weight w can be added to the set of sums without collision."""
    for s in sums:
        if (s + w) in sums:
            return False
    return True

def _add_weight(sums, w):
    """Add weight w to the set of sums, returning the new set."""
    return sums | {s + w for s in sums}

def construct():
    start_time = time.time()
    deadline = start_time + 78.0  # safety margin before 90s kill
    
    # Pre-sort indices by weight descending for better pruning (large weights spread sums)
    indices_desc = list(range(1, 23))
    indices_desc.sort(key=lambda i: _WEIGHTS[i], reverse=True)
    
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback size 9
    
    # Phase 1: DFS from high cardinality down, using descending weights
    for target_size in range(22, 9, -1):
        if time.time() > deadline:
            break
        found = _dfs_find(indices_desc, target_size, start_time, deadline)
        if found is not None:
            best = found
            break
    
    # Phase 2: Randomized Greedy Construction
    if time.time() < deadline - 10:
        rand_best = _randomized_greedy(start_time, deadline - 10)
        if rand_best and len(rand_best) > len(best):
            best = rand_best
    
    # Phase 3: Try to grow the solution via local search
    if time.time() < deadline - 5:
        improved = _local_search_grow(best, start_time, deadline)
        if improved is not None and len(improved) > len(best):
            best = improved
    
    return best

def _randomized_greedy(start_time, deadline):
    rng = random.Random(123)
    best_subset = []
    best_size = 0
    
    while time.time() < deadline:
        current = []
        sums = {0}
        available = list(range(1, 23))
        rng.shuffle(available)
        
        for idx in available:
            w = _WEIGHTS[idx]
            if _can_add(sums, w):
                sums = _add_weight(sums, w)
                current.append(idx)
        
        if len(current) > best_size:
            best_size = len(current)
            best_subset = list(current)
            
        # Early exit if we found a large set
        if best_size >= 14: 
            break
            
    return best_subset

def _dfs_find(all_indices, target_size, start_time, deadline):
    """Find a subset of size target_size with all subset sums distinct."""
    n = len(all_indices)
    
    def dfs(start_idx, current, sums):
        if time.time() > deadline:
            return None
        if len(current) == target_size:
            return list(current)
        
        remaining = target_size - len(current)
        if n - start_idx < remaining:
            return None
        
        for i in range(start_idx, n):
            idx = all_indices[i]
            w = _WEIGHTS[idx]
            
            if _can_add(sums, w):
                new_sums = _add_weight(sums, w)
                current.append(idx)
                result = dfs(i + 1, current, new_sums)
                current.pop()
                if result is not None:
                    return result
        
        return None
    
    return dfs(0, [], {0})

def _local_search_grow(initial_subset, start_time, deadline):
    """Try to grow the subset by adding elements one at a time, with randomization."""
    current = sorted(initial_subset)
    current_sums = {0}
    for i in current:
        w = _WEIGHTS[i]
        current_sums = _add_weight(current_sums, w)
    
    available = [i for i in range(1, 23) if i not in current]
    
    # Try to add elements greedily, then use random restarts
    best_size = len(current)
    best_subset = list(current)
    
    # Greedy add
    for i in available:
        w = _WEIGHTS[i]
        if _can_add(current_sums, w):
            current_sums = _add_weight(current_sums, w)
            current.append(i)
            best_size = len(current)
            best_subset = list(current)
    
    # Randomized local search: try swaps to enable further additions
    remaining_time = deadline - time.time()
    if remaining_time < 2:
        return best_subset if best_size > len(initial_subset) else None
    
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline - 2:
            break
        
        # Start from best_subset
        curr = sorted(best_subset)
        curr_sums = {0}
        for i in curr:
            w = _WEIGHTS[i]
            curr_sums = _add_weight(curr_sums, w)
        
        # Try random swaps: remove one element, try to add another
        if not curr:
            continue
        removed = rng.choice(curr)
        w_rem = _WEIGHTS[removed]
        
        # Remove w_rem from sums: need to recompute
        curr.remove(removed)
        curr_sums = {0}
        for i in curr:
            w = _WEIGHTS[i]
            curr_sums = _add_weight(curr_sums, w)
        
        # Try to add the removed element back or a different one
        candidates = [i for i in range(1, 23) if i not in curr]
        rng.shuffle(candidates)
        
        improved = False
        for i in candidates:
            w = _WEIGHTS[i]
            if _can_add(curr_sums, w):
                curr_sums = _add_weight(curr_sums, w)
                curr.append(i)
                improved = True
                break
        
        if improved and len(curr) > best_size:
            best_size = len(curr)
            best_subset = list(curr)
    
    return best_subset if best_size > len(initial_subset) else None
