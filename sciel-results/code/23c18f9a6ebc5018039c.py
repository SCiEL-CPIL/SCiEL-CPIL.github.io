"""Rewritten solver: maximize cardinality of S ⊆ [1,192] (|S|≤48) such that
no selected integer divides the sum of any 3 distinct OTHER selected integers."""

import time
import random
from itertools import combinations

# Global deadline guard (evaluator kills after 90s; we use a safety margin)
_DEADLINE = 85.0
_START = time.time()

def _time_left():
    return _DEADLINE - (time.time() - _START)

def _check_validity(selected):
    """Return True if 'selected' satisfies the condition."""
    n = len(selected)
    if n < 3:
        return True
    for i, a in enumerate(selected):
        others = [selected[j] for j in range(n) if j != i]
        if len(others) < 3:
            continue
        for t in combinations(others, 3):
            if sum(t) % a == 0:
                return False
    return True

def _is_safe_to_add(selected, x):
    """Check if adding x to valid set 'selected' keeps it valid.
    Only need to check triples involving x, plus x's own divisibility against triples of selected."""
    n = len(selected)
    # Check 1: For each a in selected, does a divide (x + u + v) for any distinct u,v in selected\{a}?
    # Equivalently: does a divide x + s where s = u+v for distinct u,v in selected, u≠v, and a not in {u,v}?
    # Precompute pair sums for selected
    pair_sums = {}
    for i in range(n):
        for j in range(i+1, n):
            s = selected[i] + selected[j]
            pair_sums.setdefault(s, []).append((i, j))
    
    for i, a in enumerate(selected):
        # Need to check if a divides x + (u+v) for any pair (u,v) in selected where u≠v and a not in {u,v}
        # Iterate over all pair sums
        for s, indices in pair_sums.items():
            for (pi, pj) in indices:
                if pi == i or pj == i:
                    continue  # pair includes a
                if (x + s) % a == 0:
                    return False
    
    # Check 2: Does x divide the sum of any 3 distinct elements in selected?
    if n >= 3:
        for t in combinations(selected, 3):
            if sum(t) % x == 0:
                return False
    
    return True

def _greedy(order):
    """Greedy: add numbers in given order, checking incremental validity."""
    selected = []
    for x in order:
        if _time_left() < 1.0:
            break
        if _is_safe_to_add(selected, x):
            selected.append(x)
    return selected

def _local_search_add(initial, max_iter=200):
    """Try to add more elements to initial set."""
    selected = list(initial)
    sel_set = set(selected)
    
    for _ in range(max_iter):
        if _time_left() < 2.0:
            break
        improved = False
        for x in range(1, 193):
            if x in sel_set:
                continue
            if _is_safe_to_add(selected, x):
                selected.append(x)
                sel_set.add(x)
                improved = True
                break
        if not improved:
            break
    return selected

def construct():
    global _START
    _START = time.time()
    
    best = []
    best_size = 0
    
    # Deterministic greedy: largest first
    cand = _greedy(range(192, 0, -1))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Deterministic greedy: smallest first
    cand = _greedy(range(1, 193))
    if len(cand) > best_size:
        best = cand
        best_size = len(cand)
    
    # Try local search additions on best
    if _time_left() > 3.0:
        improved = _local_search_add(best, max_iter=50)
        if len(improved) > best_size:
            best = improved
            best_size = len(improved)
    
    # Randomized restarts
    random.seed(42)
    for restart in range(200):
        if _time_left() < 5.0:
            break
        
        nums = list(range(1, 193))
        random.shuffle(nums)
        
        selected = _greedy(nums)
        
        if len(selected) > best_size:
            best = selected
            best_size = len(selected)
        
        # Try local search on this candidate if it's promising
        if _time_left() > 5.0 and len(selected) >= best_size - 1:
            improved = _local_search_add(selected, max_iter=30)
            if len(improved) > best_size:
                best = improved
                best_size = len(improved)
    
    return best

# Quick self-test
if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
    print(f"Valid: {_check_validity(res)}")
    print(res)
