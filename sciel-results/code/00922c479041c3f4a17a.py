"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations
import random

def _count_violations_for_new(x, selected):
    """Count how many (a, triplet) pairs would be violated if x is added."""
    violations = 0
    # Check if x divides sum of any 3 distinct others in selected
    for t in combinations(selected, 3):
        if sum(t) % x == 0:
            violations += 1
    # Check if any existing a divides sum of 3 distinct others including x
    for a in selected:
        others = [v for v in selected if v != a]
        # Need to check triplets from others that include x
        # Actually, we need triplets of 3 distinct elements from selected that don't include a, and one of them is x
        # So pick 2 from others (excluding a), and x
        for pair in combinations(others, 2):
            s = pair[0] + pair[1] + x
            if s % a == 0:
                violations += 1
    return violations

def _is_valid(selected):
    """Check if the set is valid."""
    for a in selected:
        others = [v for v in selected if v != a]
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _greedy_with_safety():
    """Greedy that picks the candidate with fewest new violations."""
    selected = []
    remaining = set(range(1, 193))
    
    while remaining and len(selected) < 48:
        best_x = None
        best_violations = float('inf')
        
        for x in sorted(remaining, reverse=True):
            v = _count_violations_for_new(x, selected)
            if v < best_violations:
                best_violations = v
                best_x = x
            if best_violations == 0:
                break
        
        if best_violations > 0:
            # Try to find any with 0 violations
            for x in sorted(remaining, reverse=True):
                v = _count_violations_for_new(x, selected)
                if v == 0:
                    best_x = x
                    best_violations = 0
                    break
        
        if best_x is not None and best_violations == 0:
            selected.append(best_x)
            remaining.remove(best_x)
        else:
            break
    
    return selected

def _local_search(selected, time_limit=5.0):
    """Try to improve the set by swapping elements."""
    if not _is_valid(selected):
        return selected
    
    start = time.time()
    best = list(selected)
    best_size = len(best)
    
    for _ in range(1000):
        if time.time() - start > time_limit:
            break
        
        # Try adding a random element
        candidates = [x for x in range(1, 193) if x not in best]
        if not candidates:
            break
        
        random.shuffle(candidates)
        improved = False
        
        for x in candidates:
            new_set = best + [x]
            if _is_valid(new_set):
                best = new_set
                best_size += 1
                improved = True
                break
        
        if not improved:
            # Try swapping
            if len(best) >= 2:
                for i in range(len(best)):
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
    deadline = start + 55.0  # 5s safety margin
    
    # Start with greedy
    result = _greedy_with_safety()
    
    # Try multiple greedy strategies
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        # Try different orderings
        random.seed(attempt)
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        selected = []
        for x in candidates:
            if time.time() > deadline - 10:
                break
            if len(selected) >= 48:
                break
            if _count_violations_for_new(x, selected) == 0:
                selected.append(x)
        
        if len(selected) > len(result):
            result = selected
    
    # Local search to improve
    if time.time() < deadline - 5:
        result = _local_search(result, time_limit=3.0)
    
    return result
# EVOLVE-BLOCK-END
