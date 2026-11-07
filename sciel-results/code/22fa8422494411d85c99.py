"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

MAX_SUM = 256 * 30  # Maximum possible subset sum for 30 elements each <= 256
MAX_ELEMENTS = 30

def _precompute_divisibility():
    """
    Precompute for each sum s, the set of multiples of s up to MAX_SUM.
    Also precompute a fast divisibility check table.
    """
    # divisors[s] = set of all proper divisors of s (d < s, d|s)
    divisors = [set() for _ in range(MAX_SUM + 1)]
    for d in range(1, MAX_SUM + 1):
        for multiple in range(2 * d, MAX_SUM + 1, d):
            divisors[multiple].add(d)
    return divisors

def _get_subset_sums(elements):
    """Get all nonempty subset sums as a set."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            new_sums.add(s + e)
        sums = sums | new_sums | {e}
    return sums

def _is_divisibility_free(sums, divisors):
    """
    Check if no two distinct sums in the set divide each other.
    For each pair (a, b) with a < b, check if a divides b.
    """
    sorted_sums = sorted(sums)
    n = len(sorted_sums)
    for i in range(n):
        a = sorted_sums[i]
        if a > MAX_SUM:
            continue
        for j in range(i + 1, n):
            b = sorted_sums[j]
            if b > MAX_SUM:
                break
            if a in divisors[b]:
                return False
    return True

def _check_addition(current_sums, new_elem, divisors):
    """
    Check if adding new_elem to current set maintains validity.
    Returns new sums set if valid, None otherwise.
    """
    new_sums = set(current_sums)
    new_sums.add(new_elem)
    for s in current_sums:
        new_sums.add(s + new_elem)
    
    return _is_divisibility_free(new_sums, divisors) and new_sums or None

def _greedy_construct(elements, divisors, deadline, strategy='smallest'):
    """
    Greedily grow the set. strategy determines how to pick next element.
    """
    current = list(elements)
    current_sums = _get_subset_sums(current)
    
    if not _is_divisibility_free(current_sums, divisors):
        return current
    
    for _ in range(MAX_ELEMENTS - len(current)):
        if time.time() > deadline:
            break
        
        # Find all valid candidates
        valid = []
        for val in range(1, 257):
            if val in current:
                continue
            new_sums = _check_addition(current_sums, val, divisors)
            if new_sums is not None:
                valid.append((val, new_sums))
        
        if not valid:
            break
        
        # Pick based on strategy
        if strategy == 'smallest':
            val, new_sums = min(valid, key=lambda x: x[0])
        elif strategy == 'largest':
            val, new_sums = max(valid, key=lambda x: x[0])
        elif strategy == 'random':
            val, new_sums = random.choice(valid)
        elif strategy == 'sparsest':
            # Pick the one that results in fewest new sums
            val, new_sums = min(valid, key=lambda x: len(x[1]) - len(current_sums))
        else:
            val, new_sums = random.choice(valid)
        
        current.append(val)
        current_sums = new_sums
    
    return current

def _local_search(elements, divisors, deadline, max_iters=100):
    """
    Local search via swaps: try replacing one element with another to improve size.
    We aim to find a larger valid set by swapping elements.
    """
    current = list(elements)
    current_sums = _get_subset_sums(current)
    
    if not _is_divisibility_free(current_sums, divisors):
        return current
    
    best = current[:]
    best_sums = current_sums
    
    for iteration in range(max_iters):
        if time.time() > deadline:
            break
        
        improved = False
        
        # Try removing one element and adding a different one
        for i, elem in enumerate(current):
            if time.time() > deadline:
                break
            
            # Remove elem
            reduced = current[:i] + current[i+1:]
            reduced_sums = _get_subset_sums(reduced)
            
            # Try adding various new elements
            for new_val in range(1, 257):
                if new_val in reduced:
                    continue
                new_set = reduced + [new_val]
                new_sums = _get_subset_sums(new_set)
                
                if _is_divisibility_free(new_sums, divisors):
                    # This swap is valid; check if it helps future growth
                    # For now, accept any valid swap and try to grow
                    trial = _greedy_construct(new_set, divisors, deadline, strategy='random')
                    if len(trial) > len(best):
                        best = trial
                        best_sums = _get_subset_sums(trial)
                        current = trial[:]
                        current_sums = best_sums
                        improved = True
                        break
                if time.time() > deadline:
                    break
        
        if not improved:
            break
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    # Precompute divisibility structure
    divisors = _precompute_divisibility()
    
    best = []
    
    # Strategy 1: Greedy with different ordering heuristics
    strategies = ['smallest', 'largest', 'sparsest', 'random']
    for strat in strategies:
        if time.time() > deadline:
            break
        result = _greedy_construct([], divisors, deadline, strategy=strat)
        if len(result) > len(best):
            best = result
    
    # Strategy 2: Multiple random restarts with greedy
    for attempt in range(30):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 7919 + 13)
        
        # Random starting set of size 1-3
        start_size = random.randint(1, 3)
        start_elems = random.sample(range(2, 257), start_size)
        start_elems.sort()
        
        # Verify initial set is valid
        start_sums = _get_subset_sums(start_elems)
        if not _is_divisibility_free(start_sums, divisors):
            continue
        
        result = _greedy_construct(start_elems, divisors, deadline, strategy='random')
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Local search from best found so far
    if len(best) > 0:
        if time.time() < deadline - 5:
            result = _local_search(best, divisors, deadline, max_iters=50)
            if len(result) > len(best):
                best = result
    
    # Strategy 4: More diverse random restarts if time permits
    for attempt in range(50, 150):
        if time.time() > deadline:
            break
        
        random.seed(attempt * 104729 + 7)
        
        # Try different starting points
        start_size = random.randint(1, 4)
        start_elems = random.sample(range(1, 257), start_size)
        start_elems.sort()
        
        start_sums = _get_subset_sums(start_elems)
        if not _is_divisibility_free(start_sums, divisors):
            continue
        
        # Alternate between strategies
        strat = strategies[attempt % len(strategies)]
        result = _greedy_construct(start_elems, divisors, deadline, strategy=strat)
        if len(result) > len(best):
            best = result
    
    # Final local search polish
    if len(best) > 0 and time.time() < deadline - 3:
        result = _local_search(best, divisors, deadline, max_iters=30)
        if len(result) > len(best):
            best = result
    
    # Ensure we return a list of Python ints
    return [int(x) for x in best]
# EVOLVE-BLOCK-END
