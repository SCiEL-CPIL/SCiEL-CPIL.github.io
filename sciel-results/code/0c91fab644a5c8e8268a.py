"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def _check_valid(S, target=200):
    """Check if no subset of S sums to target."""
    S = sorted(S)
    n = len(S)
    # Use DP: can we form sum 'target' using elements of S?
    # Since elements are positive and distinct, use bitset DP
    dp = 1  # bit i set means sum i is achievable
    for x in S:
        dp |= (dp << x)
        if dp >> target & 1:
            return False
    return True

def construct():
    """
    Returns a nonempty subset of [1, 100] with no subset summing to 200,
    maximizing cardinality.
    
    Strategy: Try to find a set larger than 35 by adding small elements 
    while removing conflicting large elements. Use a greedy approach with 
    verification, plus a local search to try to improve.
    """
    deadline = time.time() + 50  # safety margin
    
    # Start with the known optimal candidate: {66, ..., 100}
    best_set = list(range(66, 101))
    best_size = len(best_set)
    
    # Try to add elements from [1, 65] one at a time, removing minimal conflicts
    # For each candidate x in [1, 65], check if we can add it while maintaining validity
    # by removing elements that would create a sum of 200
    
    for x in range(1, 66):
        if time.time() > deadline:
            break
        
        # Try adding x to best_set
        trial = best_set + [x]
        if _check_valid(trial, 200):
            best_set = trial
            best_size += 1
    
    # Now try to add more elements greedily
    for x in range(1, 66):
        if time.time() > deadline:
            break
        if x in best_set:
            continue
        trial = best_set + [x]
        if _check_valid(trial, 200):
            best_set = trial
            best_size += 1
    
    # Local search: try swapping elements to increase size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        current_set = set(best_set)
        for x in range(1, 66):
            if time.time() > deadline:
                break
            if x in current_set:
                continue
            # Try adding x and removing one element to keep same or better size
            for y in list(current_set):
                if time.time() > deadline:
                    break
                new_set = (current_set - {y}) | {x}
                if len(new_set) > len(best_set):
                    if _check_valid(list(new_set), 200):
                        best_set = list(new_set)
                        best_size = len(best_set)
                        improved = True
                        break
    
    return best_set

# EVOLVE-BLOCK-END
