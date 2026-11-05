"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

N = 256

# EVOLVE-BLOCK-START

def _is_valid_sidon(selected):
    """Check if selected is a valid Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            b = selected[j]
            s = a + b
            if s in sums:
                return False
            sums.add(s)
    return True

def _is_maximal(selected):
    """Check if selected is inclusion-maximal in [1, N]."""
    selected_set = set(selected)
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Try adding x
        new_sums = set(sums)
        valid = True
        for a in selected:
            s = a + x
            if s in new_sums:
                valid = False
                break
            new_sums.add(s)
        if valid:
            s = 2 * x
            if s in new_sums:
                valid = False
            else:
                new_sums.add(s)
        if valid:
            return False
    return True

def _greedy_random(seed):
    """Greedy random construction with random ordering."""
    random.seed(seed)
    elements = list(range(1, N + 1))
    random.shuffle(elements)
    
    selected = []
    sums = set()
    
    for x in elements:
        added = [x + a for a in selected] + [2 * x]
        # Check for internal collisions
        if len(set(added)) != len(added):
            continue
        # Check against existing sums
        if sums.intersection(added):
            continue
        selected.append(x)
        sums.update(added)
    
    return selected

def _local_search(selected, max_iterations=10000):
    """Try to reduce size by removing elements while maintaining maximality."""
    if not _is_valid_sidon(selected):
        return selected
    
    selected = list(selected)
    
    for _ in range(max_iterations):
        improved = False
        for i in range(len(selected)):
            # Try removing selected[i]
            candidate = selected[:i] + selected[i+1:]
            if not candidate:
                continue
            if _is_valid_sidon(candidate) and _is_maximal(candidate):
                selected = candidate
                improved = True
                break
        if not improved:
            break
    
    return selected

def construct():
    """Construct a small inclusion-maximal Sidon set in [1, 256]."""
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    best = None
    best_size = float('inf')
    
    # First, get a baseline with the original greedy (ascending)
    selected, sums = [], set()
    for x in range(1, N + 1):
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    baseline = _local_search(list(selected))
    best = baseline
    best_size = len(baseline)
    
    # Try multiple random greedy constructions
    seed = 42
    while time.time() < deadline:
        candidate = _greedy_random(seed)
        if len(candidate) < best_size:
            candidate = _local_search(candidate)
            if len(candidate) < best_size:
                best = candidate
                best_size = len(candidate)
        seed += 1
        
        # Periodically check time
        if seed % 100 == 0 and time.time() > deadline:
            break
    
    return best

# EVOLVE-BLOCK-END
