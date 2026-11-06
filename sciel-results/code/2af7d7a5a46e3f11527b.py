import time
import random
from fractions import Fraction

_DEADLINE = 75.0
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

def _try_greedy_float(order):
    """
    Greedy construction using float arithmetic.
    """
    sums = [0.0]
    sums_set = {0.0}
    chosen = []
    
    for a in order:
        r = 1.0 / a
        new_sums = [s + r for s in sums]
        
        # Check collision
        collides = False
        for ns in new_sums:
            if ns in sums_set:
                collides = True
                break
        
        if collides:
            continue
            
        sums.extend(new_sums)
        sums_set.update(new_sums)
        chosen.append(a)
        
    return chosen

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy_float(order)
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand

    # Verify best with Fraction
    if best_k > 0:
        sums = {Fraction(0)}
        valid = True
        for a in best:
            r = Fraction(1, a)
            new_sums = [s + r for s in sums]
            for ns in new_sums:
                if ns in sums:
                    valid = False
                    break
            if not valid:
                break
            sums.update(new_sums)
            
        if valid:
            return list(best)
        else:
            return list(seed)
            
    return list(seed)
