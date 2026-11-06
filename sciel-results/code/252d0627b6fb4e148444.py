"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

_DEADLINE = 75.0

def _compute_lcm_1_22():
    return reduce(lcm, range(1, 23))

_L = _compute_lcm_1_22()

def _is_dissociated(subset):
    if not subset:
        return True
    vals = [_L // a for a in subset]
    sums = {0}
    for v in vals:
        new_sums = set()
        for s in sums:
            ns = s + v
            if ns in sums:
                return False
            new_sums.add(ns)
        if new_sums & sums:
            return False
        sums.update(new_sums)
    return True

def _search():
    start_time = time.time()
    best = []
    vals = {a: _L // a for a in range(1, 23)}
    elements = list(range(1, 23))
    
    def backtrack(current, sums_set, idx):
        nonlocal best
        if time.time() - start_time > _DEADLINE:
            return
        if len(current) > len(best):
            best = list(current)
        if idx >= len(elements):
            return
        c = elements[idx]
        v = vals[c]
        new_sums = set()
        collision = False
        for s in sums_set:
            ns = s + v
            if ns in sums_set:
                collision = True
                break
            new_sums.add(ns)
        if not collision and new_sums & sums_set:
            collision = True
        if not collision:
            current.append(c)
            backtrack(current, sums_set | new_sums, idx + 1)
            current.pop()
        backtrack(current, sums_set, idx + 1)
    
    backtrack([], {0}, 0)
    return best

def construct():
    result = _search()
    if result and len(result) >= 9:
        if _is_dissociated(result):
            return result
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
