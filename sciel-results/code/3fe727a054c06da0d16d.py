"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _count_violations_for_new(x, selected):
    """Count how many (a, triplet) pairs would be violated if x is added."""
    violations = 0
    n = len(selected)
    if n >= 3:
        # Check if x divides sum of any 3 distinct others in selected
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    if (selected[i] + selected[j] + selected[k]) % x == 0:
                        violations += 1
    # Check if any existing a divides sum of 3 distinct others including x
    if n >= 2:
        for a in selected:
            # Pick 2 from selected excluding a, and x
            for i in range(n):
                for j in range(i+1, n):
                    if i == selected.index(a) and j == selected.index(a):
                        continue
                    # We need to pick 2 distinct elements from selected that are not a
                    # Let's iterate properly
                    pass
            # Better approach: iterate over pairs from selected where both are not a
            for i in range(n):
                if selected[i] == a:
                    continue
                for j in range(i+1, n):
                    if selected[j] == a:
                        continue
                    s = selected[i] + selected[j] + x
                    if s % a == 0:
                        violations += 1
    return violations

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

def _greedy_largest_first():
    """Greedy that picks the largest valid candidate first."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        for x in sorted(remaining, reverse=True):
            if _count_violations_for_new(x, selected) == 0:
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
            if _count_violations_for_new(x, selected) == 0:
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
            if _count_violations_for_new(x, selected) == 0:
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
            new_set = best + [x]
            if _is_valid(new_set):
                best = new_set
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try swapping: remove one, add one
            if len(best) >= 2:
                for i in range(len(best)):
                    if time.time() - start > time_limit:
                        break
                    for x in candidates:
                        if x in best:
                            continue
                        new_set = best[:i] + best[i+1:] + [x]
                        if _is_valid(new_set):
                            best = new_set
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
