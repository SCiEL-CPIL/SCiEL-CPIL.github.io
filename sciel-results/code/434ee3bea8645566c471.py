"""Rewritten program: model-and-optimize approach for large pairwise noncoprime sets."""
import time
import math
import random
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    """
    Return a nonempty subset of [1,255] containing 255, with gcd(a,b)>1 for all distinct pairs.
    Maximize cardinality.
    
    Strategy:
    1. The set of all multiples of 3 in [1,255] is a valid solution of size 85.
    2. We attempt to improve upon this via local search (add, remove, swap operations)
       using precomputed incompatibility matrix.
    3. We also try starting from other prime multiples and mixed strategies.
    """
    start_time = time.time()
    deadline = 85  # Safety margin before 90s kill

    # Candidates: all numbers in [1,255] that share a factor with 255 (i.e., divisible by 3, 5, or 17)
    candidates = []
    for i in range(1, 256):
        if i % 3 == 0 or i % 5 == 0 or i % 17 == 0:
            candidates.append(i)
    
    n = len(candidates)
    cand_idx = {v: i for i, v in enumerate(candidates)}
    
    # Precompute incompatibility: incompat[i][j] = True if gcd(candidates[i], candidates[j]) == 1
    incompat = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if math.gcd(candidates[i], candidates[j]) == 1:
                incompat[i][j] = True
                incompat[j][i] = True
    
    # Index of 255
    idx_255 = cand_idx[255]
    
    def is_compatible_with_all(x, selected):
        """Check if x is compatible with all elements in selected (set of indices)"""
        for y in selected:
            if incompat[x][y]:
                return False
        return True
    
    def try_additions(current):
        """Try to add elements to current set. Returns (new_set, improved)"""
        improved = False
        for x in range(n):
            if x in current:
                continue
            if is_compatible_with_all(x, current):
                current = current | {x}
                improved = True
        return current, improved
    
    def try_single_swap(current):
        """Try removing one element and adding others. Returns (new_set, improved)"""
        for x in list(current):
            if x == idx_255:
                continue
            test_set = current - {x}
            added = 0
            for y in range(n):
                if y in test_set:
                    continue
                if is_compatible_with_all(y, test_set):
                    test_set = test_set | {y}
                    added += 1
            if len(test_set) > len(current):
                return test_set, True
        return current, False
    
    # Strategy 1: All multiples of 3
    mult3 = set(i for i, v in enumerate(candidates) if v % 3 == 0)
    
    # Strategy 2: All multiples of 5
    mult5 = set(i for i, v in enumerate(candidates) if v % 5 == 0)
    
    # Strategy 3: All multiples of 17
    mult17 = set(i for i, v in enumerate(candidates) if v % 17 == 0)
    
    # Strategy 4: M2 = elements divisible by at least 2 of {3,5,17}
    M2 = set(i for i, v in enumerate(candidates) 
             if (v % 3 == 0 and v % 5 == 0) or (v % 3 == 0 and v % 17 == 0) or (v % 5 == 0 and v % 17 == 0))
    
    # Initialize with best starting set
    initial_sets = [mult3, mult5, mult17, M2]
    best_size = 0
    best_set = mult3
    for s in initial_sets:
        if len(s) > best_size:
            best_size = len(s)
            best_set = s
    
    # Local search from best initial set
    current = set(best_set)
    
    improved = True
    iteration = 0
    while improved and time.time() - start_time < deadline:
        iteration += 1
        current_size = len(current)
        
        # Try additions
        current, improved = try_additions(current)
        if improved:
            continue
        
        # Try single swaps
        current, improved = try_single_swap(current)
        if improved:
            continue
        
        # Try 2-for-1 swaps: remove 2 elements, add 3+
        if time.time() - start_time < deadline - 5:
            found_swap = False
            current_list = list(current)
            for i in range(len(current_list)):
                for j in range(i + 1, len(current_list)):
                    if current_list[i] == idx_255 or current_list[j] == idx_255:
                        continue
                    test_set = current - {current_list[i], current_list[j]}
                    added = 0
                    for y in range(n):
                        if y in test_set:
                            continue
                        if is_compatible_with_all(y, test_set):
                            test_set = test_set | {y}
                            added += 1
                    if len(test_set) > len(current):
                        current = test_set
                        improved = True
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if not improved and not found_swap:
                break
    
    # Try multiple random restarts with different initial sets
    if time.time() - start_time < deadline - 10:
        random.seed(42)
        for restart in range(5):
            if time.time() - start_time > deadline - 10:
                break
            
            # Create a random subset of candidates containing 255
            # Start with multiples of 3, then randomly remove some
            current = set(mult3)
            # Remove 10-20% randomly
            to_remove = random.sample(list(current - {idx_255}), min(15, len(current) - 1))
            for x in to_remove:
                current.discard(x)
            
            # Add random compatible elements
            for x in range(n):
                if x not in current and is_compatible_with_all(x, current):
                    current.add(x)
            
            # Local search
            improved = True
            while improved and time.time() - start_time < deadline - 10:
                current, improved = try_additions(current)
                if improved:
                    continue
                current, improved = try_single_swap(current)
                if improved:
                    continue
                break
            
            if len(current) > len(best_set):
                best_set = current
    
    # Convert back to values
    result = sorted(candidates[i] for i in best_set)
    
    # Ensure 255 is in the result
    if 255 not in result:
        result.append(255)
    
    return result

# EVOLVE-BLOCK-END
