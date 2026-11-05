"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import sys
from math import gcd
from functools import reduce

_DEADLINE = None
_BEST = None

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 85.0  # Safety margin (90s limit)
    _BEST = None

def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    if not lst:
        return 1
    return reduce(_lcm, lst)

def _can_form_target_int(target_num, elements, lcm_val):
    """
    Check if target_num/lcm_val can be formed by a non-empty subset of {1/y for y in elements}.
    Equivalent to: can we find a subset of {lcm_val // y for y in elements} that sums to target_num?
    """
    if not elements:
        return False
    
    # Convert to integer values
    vals = [lcm_val // y for y in elements]
    target = target_num
    
    # Quick check: if sum of all vals < target, impossible
    total = sum(vals)
    if total < target:
        return False
    
    # If target equals total, then all elements must be used - check if non-empty
    if target == total:
        return True
    
    # Sort descending for better pruning
    vals = sorted(vals, reverse=True)
    n = len(vals)
    
    # Precompute suffix sums
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + vals[i]
    
    found = [False]
    
    def dfs(idx, acc):
        if found[0]:
            return
        if acc == target:
            found[0] = True
            return
        if idx == n:
            return
        if acc > target:
            return
        # Pruning: even if we take all remaining, can we reach target?
        if acc + suffix_sums[idx] < target:
            return
            
        # Option 2: Take vals[idx] first (larger values first for pruning)
        new_acc = acc + vals[idx]
        if new_acc <= target:
            dfs(idx + 1, new_acc)
        if found[0]:
            return
            
        # Option 1: Skip vals[idx]
        dfs(idx + 1, acc)
        
    dfs(0, 0)
    return found[0]

def _is_valid_set(selected):
    """Check if the set is reciprocal-subsum-free."""
    if len(selected) <= 1:
        return True
    
    lcm_val = _lcm_list(selected)
    for x in selected:
        target_num = lcm_val // x
        others = [y for y in selected if y != x]
        if _can_form_target_int(target_num, others, lcm_val):
            return False
    return True

def _greedy_decreasing():
    """
    Build set by adding elements from 24 down to 1.
    """
    selected = []
    
    for x in range(24, 0, -1):
        if selected:
            lcm_val = _lcm_list(selected)
            target_num = lcm_val // x
            if _can_form_target_int(target_num, selected, lcm_val):
                continue
        selected.append(x)
        
    return selected

def _greedy_with_order(order):
    """
    Build set by adding elements in the given order.
    """
    selected = []
    
    for x in order:
        if selected:
            lcm_val = _lcm_list(selected)
            target_num = lcm_val // x
            if _can_form_target_int(target_num, selected, lcm_val):
                continue
        selected.append(x)
        
    return selected

def _local_search(initial_set, time_limit):
    """
    Try to improve the set by local search.
    """
    best = list(initial_set)
    best_size = len(best)
    
    all_elements = set(range(1, 25))
    best_set = set(best)
    candidates = sorted(all_elements - best_set)
    
    deadline = time.time() + time_limit
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try removing each element in best and adding each candidate
        for i, remove_el in enumerate(best):
            if time.time() > deadline:
                break
            new_set = best[:i] + best[i+1:]
            new_set_set = set(new_set)
            
            for cand in candidates:
                if time.time() > deadline:
                    break
                if cand in new_set_set:
                    continue
                trial = new_set + [cand]
                if _is_valid_set(trial):
                    if len(trial) > best_size:
                        best = trial
                        best_size = len(best)
                        best_set = set(best)
                        candidates = sorted(all_elements - best_set)
                        improved = True
                        break
            if improved:
                break
    return best

def construct():
    _init()
    
    # Start with decreasing greedy
    best = _greedy_decreasing()
    
    if not best:
        best = [24]
    
    # Try a few other orderings
    orderings = [
        list(range(24, 0, -1)),  # decreasing (already done)
        list(range(1, 25)),      # increasing
        [24, 1, 23, 2, 22, 3, 21, 4, 20, 5, 19, 6, 18, 7, 17, 8, 16, 9, 15, 10, 14, 11, 13, 12],  # alternating
    ]
    
    for order in orderings[1:]:  # Skip the first one (decreasing)
        if time.time() > _DEADLINE - 10:
            break
        candidate = _greedy_with_order(order)
        if len(candidate) > len(best):
            best = candidate
    
    # Try local search with remaining time
    elapsed = time.time() - (_DEADLINE - 85.0)
    remaining = max(1.0, 85.0 - elapsed - 5.0)
    
    if len(best) < 24 and remaining > 2.0:
        improved = _local_search(best, remaining)
        if len(improved) > len(best):
            best = improved
    
    _BEST = best
    return list(_BEST)
# EVOLVE-BLOCK-END
