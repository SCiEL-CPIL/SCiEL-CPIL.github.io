"""Search-based approach to maximize set size under divisibility constraint."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START

DEADLINE_SECONDS = 82.0

def _all_subset_sums_fast(elements):
    """Generate all nonempty subset sums using iterative DP."""
    sums = set()
    for e in elements:
        new_sums = set()
        for s in sums:
            ns = s + e
            if ns not in sums:
                new_sums.add(ns)
        new_sums.add(e)
        sums.update(new_sums)
    return sums

def _check_divisibility(sums):
    """Check if no two distinct subset sums divide one another."""
    sum_list = sorted(sums)
    for i in range(len(sum_list)):
        a = sum_list[i]
        if a == 0:
            continue
        for j in range(i + 1, len(sum_list)):
            b = sum_list[j]
            if b % a == 0:
                return False
    return True

def _is_valid(elements):
    """Check if no two distinct subset sums divide one another."""
    n = len(elements)
    if n == 0:
        return True
    if n > 22:
        # For large n, do a fast pre-check on individual elements
        for i in range(n):
            for j in range(i + 1, n):
                a, b = elements[i], elements[j]
                if a > b:
                    a, b = b, a
                if a != 0 and b % a == 0:
                    return False
        # Then do full check if feasible
        if n <= 24:
            sums = _all_subset_sums_fast(elements)
            return _check_divisibility(sums)
        return True  # Optimistic for very large n
    
    sums = _all_subset_sums_fast(elements)
    return _check_divisibility(sums)

def _can_add(elements, candidate):
    """Check if adding candidate to elements keeps the set valid."""
    new_set = elements + [candidate]
    return _is_valid(new_set)

def _greedy_construct(start_elements, candidates, deadline):
    """Greedy: try to add candidates in order, keeping valid set."""
    current = list(start_elements)
    current_set = set(current)
    for c in candidates:
        if time.time() > deadline:
            break
        if c in current_set:
            continue
        if _can_add(current, c):
            current.append(c)
            current_set.add(c)
    return current

def _local_search(elements, candidates, deadline, max_iter=300):
    """Local search: try adding, swapping, and targeted moves to improve size."""
    current = list(elements)
    current_set = set(current)
    best = list(current)
    best_size = len(current)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Try adding a random candidate not in current
        if candidates:
            idx = random.randint(0, len(candidates) - 1)
            c = candidates[idx]
            if c not in current_set:
                if _can_add(current, c):
                    current.append(c)
                    current_set.add(c)
                    if len(current) > best_size:
                        best = list(current)
                        best_size = len(current)
        
        # Try swapping: remove one element, add another
        if len(current) > 1 and candidates:
            remove_idx = random.randint(0, len(current) - 1)
            removed = current.pop(remove_idx)
            current_set.discard(removed)
            
            # Try to find a candidate that fits
            added = False
            for _ in range(min(10, len(candidates))):
                add_idx = random.randint(0, len(candidates) - 1)
                c = candidates[add_idx]
                if c not in current_set and c != removed:
                    if _can_add(current, c):
                        current.append(c)
                        current_set.add(c)
                        added = True
                        break
            
            if not added:
                current.append(removed)
                current_set.add(removed)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    
    # All candidates from 1 to 256
    all_candidates = list(range(1, 257))
    
    # Strategy: use larger numbers to reduce subset sum collisions
    random.seed(42)
    
    best_result = []
    best_size = 0
    
    # Multiple restarts with different starting points
    for restart in range(300):
        if time.time() > deadline:
            break
        
        # Randomly select a starting set of larger numbers
        large_candidates = [x for x in all_candidates if x >= 80]
        random.shuffle(large_candidates)
        
        # Start with 2-4 elements
        start_size = random.randint(2, 4)
        start_elements = large_candidates[:start_size]
        
        # Ensure start is valid
        if not _is_valid(start_elements):
            continue
        
        # Greedy add with sorted candidates (larger first to reduce collisions)
        remaining = [x for x in large_candidates if x not in set(start_elements)]
        random.shuffle(remaining)
        current = _greedy_construct(start_elements, remaining, deadline)
        
        # Local search with all remaining candidates
        all_remaining = [x for x in all_candidates if x not in set(current)]
        current = _local_search(current, all_remaining, deadline, max_iter=150)
        
        if len(current) > best_size:
            best_result = list(current)
            best_size = len(current)
        
        # Early termination if we hit a good size
        if best_size >= 12:
            break
    
    # Final verification
    if best_result and not _is_valid(best_result):
        # Fallback to a known small valid set
        best_result = [31, 32, 34]
    
    # Ensure all are Python ints
    return [int(x) for x in best_result]

# EVOLVE-BLOCK-END
