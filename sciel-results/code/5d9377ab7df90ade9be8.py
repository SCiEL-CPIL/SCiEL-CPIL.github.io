"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _count_violations_for_new(selected, x):
    """Count violations if we add x to selected. Returns 0 if valid."""
    if x in selected:
        return 1
    n = len(selected)
    violations = 0
    
    # Check x as divisor: does x divide sum of any 3 from selected?
    for triple in combinations(selected, 3):
        if sum(triple) % x == 0:
            violations += 1
    
    # Check each existing a: does a divide sum of any triple that includes x?
    for i, a in enumerate(selected):
        rest = [selected[j] for j in range(n) if j != i]
        for pair in combinations(rest, 2):
            s = x + pair[0] + pair[1]
            if s % a == 0:
                violations += 1
    
    return violations

def _is_valid_set(selected):
    """Check if no element divides the sum of any 3 distinct other elements."""
    n = len(selected)
    if n < 4:
        return True
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _greedy_largest_first(deadline):
    """Simple greedy: try largest numbers first."""
    selected = []
    selected_set = set()
    
    for x in range(192, 0, -1):
        if len(selected) >= 48 or time.time() >= deadline:
            break
        if x in selected_set:
            continue
        
        v = _count_violations_for_new(selected, x)
        if v == 0:
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _greedy_with_priority(deadline):
    """Greedy that picks elements with fewest violations first."""
    selected = []
    selected_set = set()
    available = list(range(192, 0, -1))
    
    while len(selected) < 48 and time.time() < deadline:
        best_x = None
        best_violations = float('inf')
        
        for x in available:
            v = _count_violations_for_new(selected, x)
            if v == 0:
                best_x = x
                best_violations = 0
                break
            elif v < best_violations:
                best_violations = v
                best_x = x
        
        if best_x is None or best_violations > 0:
            break
        
        selected.append(best_x)
        selected_set.add(best_x)
        available.remove(best_x)
    
    return selected

def _greedy_build(order, n=192, max_size=48, deadline=None):
    """Greedy build: try to add elements in given order."""
    selected = []
    selected_set = set()
    
    for x in order:
        if len(selected) >= max_size:
            break
        if deadline is not None and time.time() > deadline:
            break
        if x in selected_set:
            continue
        
        v = _count_violations_for_new(selected, x)
        if v == 0:
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_search(selected, deadline, max_iter=200):
    """Try to improve by swapping elements."""
    selected = list(selected)
    selected_set = set(selected)
    available = [x for x in range(1, 193) if x not in selected_set]
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        
        # First try to add a new element
        for x in sorted(available, reverse=True):
            v = _count_violations_for_new(selected, x)
            if v == 0:
                selected.append(x)
                selected_set.add(x)
                available.remove(x)
                improved = True
                break
        
        # Try swapping: remove one element, add another
        if not improved:
            for i in range(len(selected)):
                if time.time() > deadline:
                    break
                
                without_i = selected[:i] + selected[i+1:]
                
                for x in sorted(available, reverse=True):
                    if x in selected_set:
                        continue
                    
                    v = _count_violations_for_new(without_i, x)
                    if v == 0:
                        selected = without_i + [x]
                        selected_set = set(selected)
                        available.remove(x)
                        available.append(selected[i])
                        improved = True
                        break
                
                if improved:
                    break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 75  # 75s safety margin (90s limit)
    
    best = []
    
    # Strategy 1: Largest-first greedy (fast, good baseline)
    result = _greedy_largest_first(deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Priority-based greedy
    if time.time() < deadline - 10:
        result = _greedy_with_priority(deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 3: Random restarts with different orderings
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline - 10:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy_build(order, deadline=deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Local search on best result
    if len(best) < 48 and time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if _is_valid_set(best):
        return best
    
    # Fallback: return largest-first result if valid
    fallback = _greedy_largest_first(deadline)
    if _is_valid_set(fallback):
        return fallback
    
    return []

# EVOLVE-BLOCK-END
