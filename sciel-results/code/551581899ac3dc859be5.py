"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def compute_sums(elements):
    """Compute subset sums for a list of elements using DP.
    Returns a list of integers (bitsets) where sums[k] has bit s set
    if s is a sum of a subset of size k.
    """
    n = len(elements)
    max_sum = sum(elements)
    # Use Python ints as bitsets. Bit i is 1 if sum i is possible.
    # sums[k] is a bitmask for subsets of size k.
    sums = [0] * (n + 1)
    sums[0] = 1  # Sum 0 is possible for size 0
    
    for x in elements:
        # Update from high cardinality to low to avoid using x multiple times
        for k in range(n - 1, -1, -1):
            # Shift sums[k] by x
            shifted = sums[k] << x
            sums[k + 1] |= shifted
            
    return sums

def is_valid(sums):
    """Check if sums for different cardinalities are disjoint."""
    # sums[0] is 1 (sum 0).
    # We need sums[k] & sums[m] == 0 for all k != m.
    # Note: sums[k] might contain sums that are also in sums[m].
    # We just check pairwise intersection.
    n = len(sums) - 1
    for k in range(n + 1):
        for m in range(k + 1, n + 1):
            if sums[k] & sums[m]:
                return False
    return True

def greedy_construct(order):
    """Greedy construction using the given order."""
    result = []
    sums = [1]  # sums[0] = 1
    
    for x in order:
        # Try adding x
        new_sums = sums[:]
        # Update DP
        for k in range(len(result), -1, -1):
            shifted = sums[k] << x
            new_sums[k + 1] |= shifted
        
        if is_valid(new_sums):
            result.append(x)
            sums = new_sums
        else:
            # Optional: if we want to be smarter, we could try to skip x
            # but standard greedy just skips.
            pass
            
    return result, sums

def local_search_improve(elements, sums, max_iterations=1000):
    """Improve the set by adding or swapping elements."""
    current_elements = list(elements)
    current_sums = sums
    current_size = len(current_elements)
    
    used = set(current_elements)
    candidates = [x for x in range(1, 257) if x not in used]
    
    improved = True
    while improved and not is_done():
        improved = False
        
        # 1. Try to add
        added = False
        for x in candidates:
            if is_done(): break
            # Try adding x
            new_sums = current_sums[:]
            for k in range(current_size, -1, -1):
                shifted = current_sums[k] << x
                new_sums[k + 1] |= shifted
            
            if is_valid(new_sums):
                current_elements.append(x)
                current_sums = new_sums
                current_size += 1
                used.add(x)
                candidates.remove(x)
                improved = True
                added = True
                break
        
        if added:
            continue
            
        # 2. Try to swap
        # For each element in current set, try swapping with a candidate
        swapped = False
        for i, x_old in enumerate(current_elements):
            if is_done(): break
            for x_new in candidates:
                if x_old == x_new: continue
                
                # Construct new set
                new_elements = current_elements[:]
                new_elements[i] = x_new
                
                # Compute sums for new set
                new_sums = compute_sums(new_elements)
                
                if is_valid(new_sums):
                    current_elements = new_elements
                    current_sums = new_sums
                    used.remove(x_old)
                    used.add(x_new)
                    candidates.remove(x_new)
                    candidates.append(x_old)
                    improved = True
                    swapped = True
                    break
            if swapped:
                break
                
    return current_elements, current_sums

def construct():
    best_set = []
    best_sums = None
    
    rng = random.Random(42)
    
    # Run multiple random greedy constructions
    num_trials = 0
    while not is_done() and num_trials < 50:
        num_trials += 1
        order = list(range(1, 257))
        rng.shuffle(order)
        
        result, sums = greedy_construct(order)
        
        if len(result) > len(best_set):
            best_set = result
            best_sums = sums
            
    # Local search on the best found
    if best_set:
        if not is_done():
            best_set, best_sums = local_search_improve(best_set, best_sums)
            
    # Ensure Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
