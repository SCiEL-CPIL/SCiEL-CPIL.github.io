"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_valid_set(selected):
    """Check if no element divides the sum of any 3 distinct other elements."""
    n = len(selected)
    sel = list(selected)
    for i, a in enumerate(sel):
        # Get all other elements
        others = [sel[j] for j in range(n) if j != i]
        # Check all triples of others
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                return False
    return True

def _count_violations(selected):
    """Count number of violating pairs (a, triple)."""
    count = 0
    n = len(selected)
    sel = list(selected)
    for i, a in enumerate(sel):
        others = [sel[j] for j in range(n) if j != i]
        for triple in combinations(others, 3):
            if sum(triple) % a == 0:
                count += 1
    return count

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
        
        # Check if adding x keeps the set valid
        candidate = selected + [x]
        cand_set = set(candidate)
        
        # Only need to check:
        # 1. x doesn't divide sum of any 3 others
        # 2. No existing element divides sum of any triple that includes x
        valid = True
        
        # Check 1: x as the divisor
        others_for_x = list(cand_set - {x})
        for triple in combinations(others_for_x, 3):
            if sum(triple) % x == 0:
                valid = False
                break
        
        if valid:
            # Check 2: each existing element a, check triples that include x
            for a in selected:
                # Triples that include x and 2 others from selected (not x, not a)
                rest = [v for v in selected if v != a]
                for pair in combinations(rest, 2):
                    s = x + pair[0] + pair[1]
                    if s % a == 0:
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    return selected

def _local_search(selected, deadline, max_iter=500):
    """Try to improve by swapping elements."""
    selected = list(selected)
    selected_set = set(selected)
    best = selected
    best_size = len(selected)
    
    # Precompute all numbers not in selected
    available = [x for x in range(1, 193) if x not in selected_set]
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        improved = False
        # Try removing one element and adding another
        for i in range(len(selected)):
            if time.time() > deadline:
                break
            
            without_i = selected[:i] + selected[i+1:]
            
            # Try to add a better element
            for x in available:
                if x in selected_set:
                    continue
                
                candidate = without_i + [x]
                cand_set = set(candidate)
                
                # Quick check
                valid = True
                # Check x as divisor
                others_for_x = list(cand_set - {x})
                for triple in combinations(others_for_x, 3):
                    if sum(triple) % x == 0:
                        valid = False
                        break
                
                if valid:
                    for a in without_i:
                        rest = [v for v in without_i if v != a]
                        for pair in combinations(rest, 2):
                            s = x + pair[0] + pair[1]
                            if s % a == 0:
                                valid = False
                                break
                        if not valid:
                            break
                
                if valid:
                    # This swap maintains size and is valid
                    # Prefer larger x to be "safer"
                    if x > selected[i]:
                        selected = candidate
                        selected_set = cand_set
                        improved = True
                        break
            
            if improved:
                break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    best = []
    
    # Strategy 1: Greedy with large-first (large numbers are safer)
    order_large = list(range(192, 0, -1))
    result = _greedy_build(order_large, deadline=deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 2: Greedy with small-first
    order_small = list(range(1, 193))
    result = _greedy_build(order_small, deadline=deadline)
    if len(result) > len(best):
        best = result
    
    # Strategy 3: Greedy with random orders
    random.seed(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        order = list(range(1, 193))
        random.shuffle(order)
        result = _greedy_build(order, deadline=deadline)
        if len(result) > len(best):
            best = result
    
    # Strategy 4: Local search on best
    if len(best) < 48 and time.time() < deadline - 5:
        improved = _local_search(best, deadline)
        if len(improved) > len(best):
            best = improved
    
    # Final validation
    if _is_valid_set(best):
        return best
    
    # Fallback: return empty or try a known valid small set
    return []

# EVOLVE-BLOCK-END
