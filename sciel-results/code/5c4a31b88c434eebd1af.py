"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import time
import sys

_DEADLINE = None
_BEST = None

def _init():
    global _DEADLINE, _BEST
    _DEADLINE = time.time() + 50.0  # Safety margin
    _BEST = None

def _can_form_target(target, elements):
    """
    Check if target can be formed by a non-empty subset of reciprocals of elements.
    elements: list of integers.
    target: Fraction.
    """
    if not elements:
        return False
    
    # Quick check: if sum of all reciprocals < target, impossible
    total_sum = sum(Fraction(1, y) for y in elements)
    if total_sum < target:
        return False
        
    # Sort elements ascending (largest reciprocals first) for better pruning
    elems = sorted(elements)
    n = len(elems)
    
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
            
        # Option 1: Skip elems[idx]
        dfs(idx + 1, acc)
        if found[0]:
            return
            
        # Option 2: Take elems[idx]
        new_acc = acc + Fraction(1, elems[idx])
        if new_acc <= target:
            dfs(idx + 1, new_acc)
            
    dfs(0, Fraction(0))
    return found[0]

def _greedy_decreasing():
    """
    Build set by adding elements from 24 down to 1.
    When adding x, only check if 1/x is formed by existing larger elements.
    Existing elements y > x cannot be violated by adding x because 1/y - 1/x < 0.
    """
    selected = []
    
    for x in range(24, 0, -1):
        # Check if 1/x is a subset sum of reciprocals of selected
        if _can_form_target(Fraction(1, x), selected):
            continue
        else:
            selected.append(x)
            
    return selected

def construct():
    _init()
    
    # The decreasing greedy is exact and fast for this structure.
    # It produces a maximal set under this ordering.
    best = _greedy_decreasing()
    
    if not best:
        best = [24]
        
    _BEST = best
    return list(_BEST)
# EVOLVE-BLOCK-END
