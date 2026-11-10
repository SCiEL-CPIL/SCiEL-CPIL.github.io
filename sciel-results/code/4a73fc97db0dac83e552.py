"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

N = 256

# EVOLVE-BLOCK-START
def _is_sidon(selected):
    sums = set()
    for a in selected:
        if 2 * a in sums:
            return False
        sums.add(2 * a)
    for a, b in combinations(selected, 2):
        s = a + b
        if s in sums:
            return False
        sums.add(s)
    return True

def _is_maximal(selected):
    selected_set = set(selected)
    sums = set()
    for a in selected:
        sums.add(2 * a)
    for a, b in combinations(selected, 2):
        sums.add(a + b)
    
    for x in range(1, N + 1):
        if x in selected_set:
            continue
        # Check if adding x breaks Sidon property
        new_sums = [x + a for a in selected] + [2 * x]
        # Check for duplicates within new_sums
        if len(set(new_sums)) != len(new_sums):
            continue
        # Check if any new sum is in existing sums
        if sums.intersection(new_sums):
            continue
        # If we reach here, x can be added, so not maximal
        return False
    return True

def _sidon_and_maximal(selected):
    return _is_sidon(selected) and _is_maximal(selected)

def _greedy_sidon(order):
    selected, sums = [], set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        if len(set(added)) == len(added) and not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    return selected

def _try_minimize_by_removal(selected):
    # Try to remove elements while maintaining maximality
    # This is a local search: if removing an element still leaves a maximal set, keep it removed
    changed = True
    while changed:
        changed = False
        for i in range(len(selected)):
            candidate = selected[:i] + selected[i+1:]
            if _sidon_and_maximal(candidate):
                selected = candidate
                changed = True
                break
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    best_size = float('inf')
    best_set = None
    
    # 1. Try standard ascending greedy
    asc_greedy = _greedy_sidon(range(1, N + 1))
    if _sidon_and_maximal(asc_greedy):
        best_set = asc_greedy
        best_size = len(asc_greedy)
    
    # 2. Try descending greedy (often yields smaller maximal sets)
    desc_greedy = _greedy_sidon(range(N, 0, -1))
    if _sidon_and_maximal(desc_greedy):
        if len(desc_greedy) < best_size:
            best_set = desc_greedy
            best_size = len(desc_greedy)
    
    # 3. Randomized greedy with local minimization
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Random shuffle
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy construction
        candidate = _greedy_sidon(order)
        
        if not _is_sidon(candidate):
            continue
            
        if not _is_maximal(candidate):
            continue
            
        # Try to minimize
        minimized = _try_minimize_by_removal(candidate)
        
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_set = minimized
            # If we found a very small set, we might stop early or continue
            if best_size <= 16: # Heuristic lower bound guess for N=256
                break
                
        # If time is running out, break
        if time.time() > deadline - 2:
            break
            
    # Fallback: if no maximal set found (should not happen), return ascending greedy
    if best_set is None:
        best_set = asc_greedy
        
    return best_set
# EVOLVE-BLOCK-END
