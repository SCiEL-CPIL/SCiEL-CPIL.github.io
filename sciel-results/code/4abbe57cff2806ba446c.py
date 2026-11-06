import time
from math import lcm

_START = time.time()
_DEADLINE = 85.0

def _time_remaining():
    return _DEADLINE - (time.time() - _START)

def _greedy_fill(base_set):
    """
    Given a base set of selected elements, greedily add more elements from [1, 24]
    that are not in base_set, processing from largest to smallest.
    """
    L = 1
    for i in range(2, 25):
        L = lcm(L, i)
    
    weights = {k: L // k for k in range(1, 25)}
    
    selected = list(base_set)
    reachable = {0}
    
    # Update reachable sums for base_set
    for k in selected:
        w = weights[k]
        new_reachable = set()
        for s in reachable:
            new_reachable.add(s + w)
        reachable |= new_reachable
        
    # Try to add remaining elements
    for k in range(24, 0, -1):
        if _time_remaining() < 1.0:
            break
        if k in selected:
            continue
            
        w = weights[k]
        if w not in reachable:
            selected.append(k)
            new_reachable = set()
            for s in reachable:
                new_reachable.add(s + w)
            reachable |= new_reachable
            
    return selected

def construct():
    if _time_remaining() < 1.0:
        return list(range(13, 25))
        
    # Initial Greedy
    S_current = _greedy_fill(set())
    
    # Local Search: Try removing one element and refilling
    improved = True
    while improved and _time_remaining() > 5.0:
        improved = False
        for y in list(S_current):
            if _time_remaining() < 5.0:
                break
            base = set(S_current) - {y}
            S_candidate = _greedy_fill(base)
            if len(S_candidate) > len(S_current):
                S_current = S_candidate
                improved = True
                # Restart the loop to try removing from the new set
                break
                
    if not S_current:
        return [13]
        
    return S_current

_RESULT = construct()

def construct():
    return _RESULT
