import time
import random
import sys

N = 256

def _check_sidon(selected):
    """Check if selected is a Sidon set. Returns True if valid."""
    sums = set()
    n = len(selected)
    for i in range(n):
        a = selected[i]
        if 2 * a in sums:
            return False
        sums.add(2 * a)
        for j in range(i + 1, n):
            s = a + selected[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _check_maximal(selected):
    """Check if selected is inclusion-maximal. Returns True if maximal."""
    selected_set = set(selected)
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Try adding x
        new_selected = list(selected) + [x]
        if _check_sidon(new_selected):
            return False
    return True

def _greedy_random_order(deadline):
    """Build a maximal Sidon set using random order greedy."""
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    selected = []
    sums = set()
    
    for x in elements:
        if time.time() > deadline:
            break
        # Check if x can be added
        added_sums = [x + a for a in selected] + [2 * x]
        # Check for duplicates in added_sums
        if len(set(added_sums)) != len(added_sums):
            continue
        # Check intersection with existing sums
        if not sums.intersection(added_sums):
            selected.append(x)
            sums.update(added_sums)
    
    return selected

def _reduce_maximal(selected, deadline):
    """Try to remove elements while preserving maximality."""
    selected = list(selected)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        # Try removing each element
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            candidate = selected[:i] + selected[i+1:]
            if _check_sidon(candidate) and _check_maximal(candidate):
                selected = candidate
                improved = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin for 60s limit
    
    best_set = None
    best_size = N + 1
    
    # First, try the original ascending greedy as a baseline
    selected_asc, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected_asc] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected_asc.append(x)
            sums.update(added)
    
    # Reduce the ascending greedy set
    reduced_asc = _reduce_maximal(selected_asc, deadline)
    if len(reduced_asc) < best_size:
        best_size = len(reduced_asc)
        best_set = reduced_asc
    
    # Now try multiple random restarts
    seed = 42
    while time.time() < deadline:
        random.seed(seed)
        seed += 1
        selected = _greedy_random_order(deadline)
        if len(selected) >= best_size:
            continue
        reduced = _reduce_maximal(selected, deadline)
        if len(reduced) < best_size:
            best_size = len(reduced)
            best_set = reduced
    
    if best_set is None:
        return selected_asc
    
    return best_set
