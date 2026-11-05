"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import itertools

def _check_validity(selected):
    """Check if selected set is reciprocal-subsum-free."""
    if not selected:
        return True
    sel_set = set(selected)
    for a in selected:
        target = Fraction(1, a)
        others = [b for b in selected if b != a]
        if not others:
            continue
        # Check if target can be formed by non-empty subset of others
        # Use DFS with pruning
        others_sorted = sorted(others)  # ascending, so largest reciprocals first
        n = len(others_sorted)
        total_sum = sum(Fraction(1, x) for x in others_sorted)
        if total_sum < target:
            continue  # can't form target
        
        found = [False]
        
        def dfs(idx, acc):
            if found[0]:
                return
            if acc == target:
                found[0] = True
                return
            if idx >= n or acc > target:
                return
            # Prune: even taking all remaining can't reach target
            remaining = sum(Fraction(1, others_sorted[i]) for i in range(idx, n))
            if acc + remaining < target:
                return
            # Option 1: include others_sorted[idx]
            dfs(idx + 1, acc + Fraction(1, others_sorted[idx]))
            # Option 2: skip
            dfs(idx + 1, acc)
        
        dfs(0, Fraction(0, 1))
        if found[0]:
            return False
    return True

def _greedy_build(order, deadline):
    """Greedily add elements in given order, checking validity."""
    selected = []
    for elem in order:
        if time.time() > deadline:
            break
        candidate = selected + [elem]
        if _check_validity(candidate):
            selected = candidate
    return selected

def construct():
    start = time.time()
    deadline = start + 50.0  # Safety margin before 60s limit
    
    # Try multiple orders and keep the best valid set
    best_set = []
    
    # Order 1: descending (large elements first, less likely to cause issues)
    orders = [
        list(range(24, 0, -1)),  # 24, 23, ..., 1
        list(range(1, 25)),      # 1, 2, ..., 24
        # Try to exclude problematic small numbers
        [x for x in range(24, 0, -1) if x > 6],
        [x for x in range(24, 0, -1) if x > 4],
        [x for x in range(24, 0, -1) if x > 2],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy_build(order, deadline)
        if len(result) > len(best_set):
            best_set = result
    
    # If we still have time, try to improve by removing one element and adding two
    if time.time() < deadline - 5.0:
        current = set(best_set)
        improved = False
        for remove_elem in list(best_set):
            if time.time() > deadline:
                break
            temp_set = [x for x in best_set if x != remove_elem]
            # Try to add elements not in current set
            for add_elem in range(1, 25):
                if add_elem in current:
                    continue
                if time.time() > deadline:
                    break
                candidate = temp_set + [add_elem]
                if _check_validity(candidate):
                    # Try to add another
                    for add_elem2 in range(1, 25):
                        if add_elem2 in set(candidate):
                            continue
                        if time.time() > deadline:
                            break
                        candidate2 = candidate + [add_elem2]
                        if _check_validity(candidate2) and len(candidate2) > len(best_set):
                            best_set = candidate2
                            current = set(best_set)
                            improved = True
                            break
    
    # Final fallback: ensure non-empty
    if not best_set:
        best_set = [24]
    
    return list(best_set)
# EVOLVE-BLOCK-END
