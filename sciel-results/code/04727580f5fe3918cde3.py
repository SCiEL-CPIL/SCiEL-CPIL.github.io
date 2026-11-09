"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _is_valid(selected):
    """Check if the set is valid."""
    n = len(selected)
    if n < 4:
        return True
    for a in selected:
        # Find all triplets from selected excluding a
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if (t[0] + t[1] + t[2]) % a == 0:
                return False
    return True

def _can_add(x, selected):
    """Check if x can be added to selected without violating constraints.
    Returns True if valid."""
    n = len(selected)
    if n < 3:
        return True
    
    # Check 1: For any a in selected, a does not divide x + y + z
    # where y, z are distinct elements in selected \ {a}
    # Note: y, z must be distinct from each other and from a.
    
    for a in selected:
        # We need to check if there exist y, z in selected \ {a} such that (x + y + z) % a == 0
        # Let rem = x % a. We need (y + z) % a == (-rem) % a.
        
        target = (-x) % a
        # Check pairs from selected excluding a
        others = [v for v in selected if v != a]
        m = len(others)
        if m < 2:
            continue
            
        # Check all pairs in others
        found = False
        for i in range(m):
            y = others[i]
            for j in range(i+1, m):
                z = others[j]
                if (x + y + z) % a == 0:
                    return False
        
    # Check 2: x does not divide a + y + z for any distinct a, y, z in selected
    # Iterate triples
    for a, y, z in combinations(selected, 3):
        if (a + y + z) % x == 0:
            return False
            
    return True

def _greedy_largest_first():
    """Greedy that picks the largest valid candidate first."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in sorted(remaining, reverse=True):
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _greedy_smallest_first():
    """Greedy that picks the smallest valid candidate first."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in sorted(remaining):
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _greedy_random(ordering_seed=0):
    """Greedy with random ordering."""
    random.seed(ordering_seed)
    selected = []
    remaining = list(range(1, 193))
    random.shuffle(remaining)
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in remaining:
            if _can_add(x, selected):
                best_x = x
                break
        
        if best_x is not None:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _local_search(selected, time_limit=10.0):
    """Try to improve the set by adding or swapping elements."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_size = len(best)
    
    for iteration in range(2000):
        if time.time() - start > time_limit:
            break
        
        candidates = [x for x in range(1, 193) if x not in best]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        # Try adding an element
        for x in candidates:
            if _can_add(x, best):
                best.append(x)
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add one
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    removed = best[i]
                    temp_best = best[:i] + best[i+1:]
                    for x in candidates:
                        if x in best:
                            continue
                        if _can_add(x, temp_best):
                            best = temp_best + [x]
                            improved = True
                            break
                    if improved:
                        break
        
        if not improved:
            break
    
    return best

def construct():
    start = time.time()
    deadline = start + 85.0  # 5s safety margin before 90s kill
    
    # Try multiple greedy strategies and keep the best
    result = []
    
    # Largest first
    if time.time() < deadline - 5:
        s = _greedy_largest_first()
        if len(s) > len(result):
            result = s
    
    # Smallest first
    if time.time() < deadline - 5:
        s = _greedy_smallest_first()
        if len(s) > len(result):
            result = s
    
    # Multiple random orderings
    for seed in range(50):
        if time.time() > deadline - 15:
            break
        s = _greedy_random(seed)
        if len(s) > len(result):
            result = s
    
    # Local search to improve
    if time.time() < deadline - 5:
        remaining_time = deadline - 5 - time.time()
        result = _local_search(result, time_limit=remaining_time)
    
    return result
# EVOLVE-BLOCK-END
