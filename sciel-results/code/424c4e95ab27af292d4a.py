"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import itertools
from collections import Counter
import time

def _max_sidon_size_fast(arr):
    """
    Compute the maximum size of a subset of arr where all unordered pair sums
    (including diagonals, i.e. 2*x for x in subset) are unique.
    Returns the maximum size.
    """
    n = len(arr)
    if n < 2:
        return n

    # Precompute pair sums for all pairs in arr
    # We use a set of tuples to represent subsets? No, that's too slow.
    # We can use backtracking.
    
    best = 0
    # Sort arr to help pruning? Not strictly necessary but might help.
    arr = sorted(arr)
    
    # For efficient checking during backtracking, we can maintain a set of used sums.
    
    def backtrack(start, current_set, used_sums):
        nonlocal best
        if len(current_set) > best:
            best = len(current_set)
        
        # If even adding all remaining elements doesn't beat best, prune
        if len(current_set) + (n - start) <= best:
            return
            
        for i in range(start, n):
            x = arr[i]
            # Check if x can be added
            # Sums to check: x + y for all y in current_set, and 2*x
            valid = True
            new_sums = []
            
            # Check 2*x
            s2 = x + x
            if s2 in used_sums:
                valid = False
            else:
                new_sums.append(s2)
            
            # Check x + y
            if valid:
                for y in current_set:
                    s = x + y
                    if s in used_sums:
                        valid = False
                        break
                    new_sums.append(s)
            
            if valid:
                # Add x
                used_sums.update(new_sums)
                current_set.append(x)
                
                backtrack(i + 1, current_set, used_sums)
                
                current_set.pop()
                for s in new_sums:
                    used_sums.remove(s)

    backtrack(0, [], set())
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin for evaluation overhead
    best_set = list(range(12))
    best_score = _max_sidon_size_fast(best_set)
    
    # Heuristic search for a set with smaller max Sidon subset
    # Strategy: Try to construct sets with many additive collisions.
    # The set [0..11] has max Sidon 4. Can we get 3?
    # A set with max Sidon 3 means no 4 distinct elements a,b,c,d in the set
    # form a Sidon set.
    
    # Random search in a smaller range might be more effective if collisions are forced.
    # But the range is fixed [0, 128].
    # Let's try random sets and evaluate.
    
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Generate random set of 12 distinct integers in [0, 128]
        s = random.sample(range(129), 12)
        
        # Quick heuristic check: if the set contains an arithmetic progression of length 4,
        # it might have a large Sidon subset? Not necessarily.
        # Let's just evaluate.
        score = _max_sidon_size_fast(s)
        
        if score < best_score:
            best_score = score
            best_set = s
            if best_score == 3:
                break # Can't get lower than 3 for n=12?
                # Actually, min Sidon size is 1. But 3 is very hard.
        
        # If we find 4, keep searching for 3.
        # If we find 3, we are very happy.
        
        # Adaptive strategy: if we haven't found better than 4, try structured sets?
        # For now, random search.
        
        if best_score == 3:
            break

    return best_set
# EVOLVE-BLOCK-END
