"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

START_TIME = time.time()
DEADLINE = 55.0  # Safety margin before 60s kill

def elapsed():
    return time.time() - START_TIME

def is_done():
    return elapsed() > DEADLINE

def try_add(x, sums, current_size):
    """Try adding element x to the set represented by sums.
    Returns new_sums if valid, else None."""
    new_sums = list(sums)
    for k in range(current_size, -1, -1):
        shifted = sums[k] << x
        candidate = new_sums[k + 1] | shifted
        # Only need to check against cardinalities m != k+1
        for m in range(current_size + 2):
            if m != k + 1 and (candidate & new_sums[m]):
                return None
        new_sums[k + 1] = candidate
    return new_sums

def greedy_construct(order):
    """Greedy: iterate through 'order', add each element if valid.
    Then do a second pass over all remaining candidates."""
    sums = [0] * (len(order) + 1)
    sums[0] = 1  # empty subset sum 0
    current_size = 0
    result = []
    used = set()
    
    # First pass
    for x in order:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    # Second pass: try all remaining candidates
    remaining = [x for x in order if x not in used]
    for x in remaining:
        if is_done():
            break
        new_sums = try_add(x, sums, current_size)
        if new_sums is not None:
            sums = new_sums
            current_size += 1
            result.append(x)
            used.add(x)
    
    return result

def backtrack_search(candidates, current_set, current_sums, current_size, best):
    """Backtracking search with forward checking.
    candidates: list of candidate values to try
    current_set: set of values already chosen
    current_sums: list of bitsets for each cardinality
    current_size: number of elements in current_set
    best: [best_result, best_sums, best_size]
    """
    if is_done():
        return
    
    # Update best if we have a larger set
    if len(current_set) > best[2]:
        best[0] = list(current_set)
        best[1] = list(current_sums)
        best[2] = len(current_set)
    
    # Try adding each candidate
    for i, x in enumerate(candidates):
        if is_done():
            return
        
        new_sums = try_add(x, current_sums, current_size)
        if new_sums is not None:
            # Recurse with remaining candidates
            backtrack_search(
                candidates[i+1:],
                current_set | {x},
                new_sums,
                current_size + 1,
                best
            )
    
    # Also try removing one element from current_set to explore alternatives
    if current_size > 0 and len(candidates) < 100:
        for x in list(current_set):
            if is_done():
                return
            
            # Remove x and try to add other candidates
            # Rebuild sums without x
            new_sums = [0] * (len(current_sums) + 1)
            for k in range(len(current_sums)):
                new_sums[k] = current_sums[k] & ~((1 << x) if k == 0 else 0)
            
            # This is complex; skip for now and rely on candidate ordering

def local_search_improve(result, sums, current_size):
    """Try to swap elements to allow adding more. Returns improved result or None."""
    if is_done():
        return None
    
    all_candidates = list(range(1, 257))
    result_set = set(result)
    candidates = [x for x in all_candidates if x not in result_set]
    
    best_result = list(result)
    best_sums = sums
    best_size = current_size
    
    for x in candidates:
        if is_done():
            break
        # Try adding x directly
        new_sums = try_add(x, best_sums, best_size)
        if new_sums is not None:
            best_sums = new_sums
            best_size += 1
            best_result.append(x)
    
    return best_result, best_sums, best_size

def construct():
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]  # Powers of 2 baseline (9 elements)
    
    # Try multiple strategies
    strategies = []
    
    # Strategy 1: Increasing order
    strategies.append(list(range(1, 257)))
    
    # Strategy 2: Decreasing order
    strategies.append(list(range(256, 0, -1)))
    
    # Strategy 3: Random orders (more of them)
    rng = random.Random(42)
    for _ in range(30):
        order = list(range(1, 257))
        rng.shuffle(order)
        strategies.append(order)
    
    # Strategy 4: Sorted by value, but with some structure
    # Try multiples of small primes first
    for prime in [2, 3, 5, 7, 11]:
        order = [x for x in range(1, 257) if x % prime == 0]
        order += [x for x in range(1, 257) if x % prime != 0]
        strategies.append(order)
    
    for order in strategies:
        if is_done():
            break
        result = greedy_construct(order)
        if len(result) > len(best_set):
            best_set = result
    
    # Backtracking search for larger sets
    # Start from the best greedy result and try to extend
    if len(best_set) < 60 and not is_done():
        best_result = list(best_set)
        best_sums = [0] * (len(best_set) + 2)
        best_sums[0] = 1
        for x in best_result:
            new_sums = try_add(x, best_sums, len(best_result) - 1)
            if new_sums is not None:
                best_sums = new_sums
                best_result.append(x)
        
        # Now try to add more using backtracking
        candidates = [x for x in range(1, 257) if x not in set(best_result)]
        backtrack_search(candidates, set(best_result), best_sums, len(best_result), [best_result, best_sums, len(best_result)])
    
    # Ensure Python ints in a list
    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
