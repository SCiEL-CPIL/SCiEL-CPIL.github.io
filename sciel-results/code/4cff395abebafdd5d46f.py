"""Improved program: Reverse greedy reduction for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
import numpy as np

N = 256
DEADLINE = time.time() + 55  # 55s safety margin (evaluator kills at 60s)

# EVOLVE-BLOCK-START

def _is_maximal(selected, sums):
    """
    Check if the Sidon set 'selected' is inclusion-maximal in [1, N].
    An element x is blocked if adding it would create a duplicate sum.
    x is blocked if 2x is in sums OR (x+a) is in sums for some a in selected.
    The set is maximal if every x in [1, N] is either in selected or blocked.
    """
    sset = set(selected)
    blocked = set()
    
    # 1. Check diagonal sums: 2x in sums => x is blocked
    for s in sums:
        if s % 2 == 0:
            x = s // 2
            if 1 <= x <= N and x not in sset:
                blocked.add(x)
                
    # 2. Check off-diagonal sums: x+a in sums for some a in selected
    for t in sums:
        for a in selected:
            x = t - a
            if 1 <= x <= N and x not in sset:
                blocked.add(x)
    
    # Check if all elements in [1, N] are either in selected or blocked
    for x in range(1, N + 1):
        if x in sset:
            continue
        if x not in blocked:
            return False
    return True

def _build_sums(selected):
    """Build the set of all unordered pair sums a+b with a<=b."""
    sums = set()
    for i in range(len(selected)):
        for j in range(i, len(selected)):
            sums.add(selected[i] + selected[j])
    return sums

def _greedy_maximal(perm):
    """
    Greedily build a maximal Sidon set by processing elements in 'perm' order.
    """
    selected = []
    sums = set()
    for x in perm:
        new_sums = {2 * x}
        valid = True
        for a in selected:
            s = x + a
            if s in new_sums or s in sums:
                valid = False
                break
            new_sums.add(s)
        
        if valid:
            selected.append(x)
            sums.update(new_sums)
            
    return selected, sums

def _reduce_maximal(selected, sums, rng):
    """
    Try to reduce the size of the maximal Sidon set while maintaining maximality.
    """
    selected = list(selected)
    sums = sums
    improved = True
    while improved:
        improved = False
        indices = list(range(len(selected)))
        rng.shuffle(indices)
        
        for idx in indices:
            x = selected[idx]
            new_selected = selected[:idx] + selected[idx+1:]
            if not new_selected:
                continue
            
            new_sums = _build_sums(new_selected)
            if _is_maximal(new_selected, new_sums):
                selected = new_selected
                sums = new_sums
                improved = True
                break
                
    return selected, sums

def construct():
    rng = random.Random()
    best_size = N + 1
    best_result = None
    
    start = time.time()
    
    while time.time() < DEADLINE - 2:
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        selected, sums = _greedy_maximal(perm)
        selected, sums = _reduce_maximal(selected, sums, rng)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_result = list(selected)
            
        if time.time() > DEADLINE - 3:
            break
            
    if best_result is None:
        perm = list(range(1, N + 1))
        selected, sums = _greedy_maximal(perm)
        best_result = list(selected)
        
    return best_result

# EVOLVE-BLOCK-END
