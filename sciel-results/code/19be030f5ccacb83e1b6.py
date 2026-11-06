"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from math import lcm
from functools import reduce

# Precompute LCM of 1..22
_L = reduce(lcm, range(1, 23))

def _is_dissociated(subset):
    """Check if all subset sums of reciprocals (scaled by LCM) are distinct."""
    if len(subset) == 0:
        return True
    vals = [_L // a for a in subset]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _backtrack_search(start_idx, current_subset, deadline):
    """
    Backtracking search to find maximum cardinality dissociated subset.
    Returns the best subset found before deadline.
    """
    best = list(current_subset)
    
    def _check_and_add(candidate_val, current_sums):
        """Try adding candidate_val to current subset. Returns new sums or None if collision."""
        new_sums = set()
        for s in current_sums:
            ns = s + candidate_val
            if ns in current_sums or ns in new_sums:
                return None
            new_sums.add(ns)
        return current_sums | new_sums
    
    def _search(idx, current_subset, current_sums):
        nonlocal best
        if time.time() > deadline:
            return
        if len(current_subset) > len(best):
            best = list(current_subset)
        
        # Try to add more elements
        for i in range(idx, 22):
            if time.time() > deadline:
                return
            val = _L // (i + 1)
            new_sums = _check_and_add(val, current_sums)
            if new_sums is not None:
                current_subset.append(i + 1)
                _search(i + 1, current_subset, new_sums)
                current_subset.pop()
    
    _search(start_idx, current_subset, {0})
    return best

def _greedy_randomized(deadline):
    """
    Randomized greedy approach with multiple restarts.
    """
    best = []
    best_len = 0
    elements = list(range(1, 23))
    
    while time.time() < deadline:
        random.shuffle(elements)
        current = []
        current_sums = {0}
        for a in elements:
            val = _L // a
            new_sums = set()
            valid = True
            for s in current_sums:
                ns = s + val
                if ns in current_sums or ns in new_sums:
                    valid = False
                    break
                new_sums.add(ns)
            if valid:
                current.append(a)
                current_sums |= new_sums
        
        if len(current) > best_len:
            best_len = len(current)
            best = list(current)
        
        # If we found a large set, try to improve with backtracking
        if best_len >= 10:
            break
    
    return best

def _local_search_improve(subset, deadline):
    """
    Try to improve a dissociated subset by swapping elements.
    """
    current = list(subset)
    current_sums = {0}
    for a in current:
        val = _L // a
        for s in list(current_sums):
            current_sums.add(s + val)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(current)):
            for j in range(1, 23):
                if j in current:
                    continue
                if time.time() > deadline:
                    return current
                
                # Try swapping current[i] with j
                new_subset = current[:]
                new_subset[i] = j
                new_subset.sort()
                
                if _is_dissociated(new_subset):
                    # Check if this actually helps (it shouldn't change size, but might enable further additions)
                    # For now, just accept it as a different valid set
                    pass
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5 second safety margin
    
    # First, try systematic backtracking from small elements
    # Strategy: try to build up from 1,2,3,...
    best = _backtrack_search(0, [], deadline)
    
    # If backtracking didn't find something large, try randomized greedy
    if len(best) < 10 and time.time() < deadline:
        greedy_result = _greedy_randomized(deadline)
        if len(greedy_result) > len(best):
            best = greedy_result
    
    # Try to improve with local search if time remains
    if time.time() < deadline - 5:
        improved = _local_search_improve(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Ensure nonempty
    if not best:
        best = [1]
    
    return best
# EVOLVE-BLOCK-END
