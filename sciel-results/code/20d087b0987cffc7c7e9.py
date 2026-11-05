"""Reciprocal dissociated set search in [1,22]."""
import time
from itertools import combinations

# Deadline guard
START_TIME = time.time()
DEADLINE = 85.0  # safety margin before 90s

def _check_time():
    return (time.time() - START_TIME) < DEADLINE

def _is_valid_subset(S):
    """
    Check if subset S of [1,22] has all reciprocal subset sums distinct.
    We check all 2^|S| subset sums of {1/x : x in S}.
    """
    n = len(S)
    if n == 0:
        return True
    # Compute all subset sums of reciprocals
    # Use a set to detect collisions
    sums = set()
    # There are 2^n subset sums
    # For efficiency, use iterative approach
    sums.add(0.0)
    for x in S:
        r = 1.0 / x
        new_sums = set()
        for s in sums:
            ns = s + r
            if ns in sums or ns in new_sums:
                return False
            new_sums.add(ns)
        sums.update(new_sums)
    return True

def _construct():
    """
    Find a large subset of [1,22] such that all reciprocal subset sums are distinct.
    Strategy: Start with a known valid set, then try to add elements greedily,
    and use local search to find larger valid sets.
    """
    # Initial known valid set (from seed)
    best = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try to add elements greedily
    all_elements = list(range(1, 23))
    current = list(best)
    
    # Try adding each remaining element
    for x in all_elements:
        if x not in current:
            test = current + [x]
            if _is_valid_subset(test):
                current = test
    
    # Local search: try swapping elements
    improved = True
    while improved and _check_time():
        improved = False
        for i in range(len(current)):
            for x in all_elements:
                if x not in current:
                    test = current[:i] + current[i+1:] + [x]
                    if len(test) == len(current) and _is_valid_subset(test):
                        # Same size but different element - might help for future additions
                        # But we want to maximize size, so only accept if it enables larger set
                        # For now, skip same-size swaps unless we're stuck
                        pass
            if not _check_time():
                break
        if not _check_time():
            break
        
        # Try adding elements again after swaps
        for x in all_elements:
            if x not in current:
                test = current + [x]
                if _is_valid_subset(test):
                    current = test + []
                    improved = True
                    break
    
    # Now try to find larger sets using backtracking
    # Start from current best and try to extend
    best_size = len(current)
    
    # Try all subsets of size best_size + 1, best_size + 2, etc.
    # But that's expensive. Instead, use a smarter approach.
    
    # Try to build up from scratch with a better ordering
    # Sort by 1/x (i.e., x descending) to make reciprocal sums more spread out
    sorted_elements = sorted(all_elements, key=lambda x: 1.0/x, reverse=True)
    
    # Greedy with different ordering
    for start_size in range(best_size + 1, min(14, best_size + 4)):
        if not _check_time():
            break
        # Try to find a valid set of size start_size
        # Use greedy: pick elements in order of largest reciprocal first
        # Then try all combinations of the remaining elements
        if start_size <= len(all_elements):
            # Try greedy construction
            greedy_set = []
            for x in sorted_elements:
                if len(greedy_set) >= start_size:
                    break
                test = greedy_set + [x]
                if _is_valid_subset(test):
                    greedy_set = test
            if len(greedy_set) == start_size:
                current = greedy_set
                best_size = start_size
                # Try to extend further
                for x in sorted_elements:
                    if x not in current:
                        test = current + [x]
                        if _is_valid_subset(test):
                            current = test
                            best_size = len(current)
    
    # Final greedy pass
    for x in sorted_elements:
        if x not in current:
            test = current + [x]
            if _is_valid_subset(test):
                current = test
    
    return current

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    result = _construct()
    if not result:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return result
