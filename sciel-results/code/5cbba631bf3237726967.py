"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations_with_replacement

def _check_valid(selected):
    """Check if all unordered triples (with repetition) have unique sums."""
    n = len(selected)
    seen = set()
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                s = selected[i] + selected[j] + selected[k]
                if s in seen:
                    return False
                seen.add(s)
    return True

def _new_sums_for_x(x, selected):
    """
    Compute all new triple sums that would be introduced by adding x to selected.
    Assumes selected is sorted and x is not in selected.
    New triples containing at least one x:
    (x, x, x)
    (x, x, y) for y in selected
    (x, y, y) for y in selected
    (x, y, z) for y < z in selected
    """
    n = len(selected)
    sums = [3 * x]
    for y in selected:
        sums.append(2 * x + y)
        sums.append(x + 2 * y)
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + selected[i] + selected[j])
    return sums

def _all_triple_sums(selected):
    """Compute all unordered triple sums for a sorted list."""
    n = len(selected)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(selected[i] + selected[j] + selected[k])
    return sums

def _greedy_construct(max_val=4096, target_size=80, time_limit=40.0):
    """
    Greedy construction: try adding smallest possible number that doesn't create a collision.
    """
    start_time = time.time()
    selected = []
    used_sums = set()
    
    for x in range(1, max_val + 1):
        if time.time() - start_time > time_limit:
            break
        if len(selected) >= target_size:
            break
        
        new_sums = _new_sums_for_x(x, selected)
        if any(s in used_sums for s in new_sums):
            continue
        
        used_sums.update(new_sums)
        selected.append(x)
    
    return selected

def _greedy_with_steps(step_base=1, max_val=4096, target_size=80, time_limit=20.0):
    """
    Greedy construction with a step pattern to spread numbers out more.
    """
    start_time = time.time()
    selected = []
    used_sums = set()
    
    x = 1
    while x <= max_val:
        if time.time() - start_time > time_limit:
            break
        if len(selected) >= target_size:
            break
        
        new_sums = _new_sums_for_x(x, selected)
        if not any(s in used_sums for s in new_sums):
            used_sums.update(new_sums)
            selected.append(x)
        
        # Increase step as we go to spread out
        step = step_base + len(selected) // 10
        x += step
    
    return selected

def _backtracking_search(initial, max_val=4096, target_size=80, time_limit=30.0, max_depth=5):
    """
    Backtracking search: try to add elements, and if stuck, backtrack and try alternatives.
    Uses heuristic ordering (smallest first) but allows backtracking.
    """
    start_time = time.time()
    best = list(initial)
    
    def search(selected, used_sums, depth, start_x):
        nonlocal best
        if time.time() - start_time > time_limit:
            return
        if len(selected) >= target_size:
            if len(selected) > len(best):
                best = list(selected)
            return
        if depth > max_depth:
            return
        
        n = len(selected)
        for x in range(start_x, max_val + 1):
            if time.time() - start_time > time_limit:
                return
            if len(selected) >= target_size:
                break
            
            new_sums = _new_sums_for_x(x, selected)
            if any(s in used_sums for s in new_sums):
                continue
            
            # Try adding x
            new_used = set(used_sums)
            new_used.update(new_sums)
            new_selected = selected + [x]
            
            if len(new_selected) > len(best):
                best = list(new_selected)
            
            # Recurse
            search(new_selected, new_used, depth + 1, x + 1)
            
            # Backtrack: remove x and try next
            # (implicitly done by loop continuing)
        
        # If we couldn't add anything, try removing the last element and trying alternatives
        # This is a simplified backtracking - in a full version, we'd try different orderings
    
    initial_sums = set(_all_triple_sums(initial))
    search(list(initial), initial_sums, 0, 1)
    
    return best

def _local_search_improve(selected, max_val=4096, time_limit=15.0):
    """
    Try to improve the set by trying to add more elements.
    """
    start_time = time.time()
    current = list(selected)
    used_sums = set(_all_triple_sums(current))
    
    for x in range(1, max_val + 1):
        if time.time() - start_time > time_limit:
            break
        if x in current:
            continue
        
        new_sums = _new_sums_for_x(x, current)
        if not any(s in used_sums for s in new_sums):
            used_sums.update(new_sums)
            current.append(x)
            current.sort()
    
    return current

def _randomized_greedy(rng, max_val=4096, target_size=80, time_limit=10.0):
    """
    Randomized greedy: try different starting points and step patterns.
    """
    start_time = time.time()
    best = []
    
    for attempt in range(50):
        if time.time() - start_time > time_limit:
            break
        
        start = rng.randint(1, 20)
        step = rng.randint(1, 3)
        
        selected = []
        used_sums = set()
        x = start
        
        while x <= max_val:
            if time.time() - start_time > time_limit:
                break
            if len(selected) >= target_size:
                break
            
            new_sums = _new_sums_for_x(x, selected)
            if not any(s in used_sums for s in new_sums):
                used_sums.update(new_sums)
                selected.append(x)
            
            x += step + rng.randint(0, 2)
        
        if len(selected) > len(best):
            best = selected
    
    return best

def construct():
    """
    Construct a set of up to 80 distinct positive integers <= 4096
    where all unordered triples (with repetition) have unique sums.
    Maximize cardinality.
    """
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Strategy 1: Simple greedy from 1
    result = _greedy_construct(max_val=4096, target_size=80, time_limit=30.0)
    best = result
    
    if time.time() > deadline:
        return best
    
    # Strategy 2: Greedy with different step patterns
    for step_base in [2, 3, 5]:
        if time.time() > deadline:
            break
        candidate = _greedy_with_steps(step_base=step_base, max_val=4096, target_size=80, time_limit=10.0)
        if len(candidate) > len(best):
            best = candidate
    
    if time.time() > deadline:
        return best
    
    # Strategy 3: Randomized greedy
    rng = random.Random(42)
    candidate = _randomized_greedy(rng, max_val=4096, target_size=80, time_limit=15.0)
    if len(candidate) > len(best):
        best = candidate
    
    if time.time() > deadline:
        return best
    
    # Strategy 4: Backtracking search from best
    if len(best) < 80:
        candidate = _backtracking_search(best, max_val=4096, target_size=80, time_limit=20.0, max_depth=3)
        if len(candidate) > len(best):
            best = candidate
    
    if time.time() > deadline:
        return best
    
    # Strategy 5: Local search improvement
    candidate = _local_search_improve(best, max_val=4096, time_limit=10.0)
    if len(candidate) > len(best):
        best = candidate
    
    return best

# EVOLVE-BLOCK-END
