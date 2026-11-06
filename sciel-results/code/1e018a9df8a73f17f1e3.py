"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from math import gcd
from functools import reduce
import random

# EVOLVE-BLOCK-START
def _lcm(a, b):
    return a * b // gcd(a, b)

def _lcm_list(lst):
    return reduce(_lcm, lst, 1)

def _is_dissociated_fast(subset):
    """Check if all subset sums of reciprocals are distinct using integer scaling."""
    if not subset:
        return True
    L = _lcm_list(subset)
    weights = [L // x for x in subset]
    n = len(weights)
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            new_sums.add(s + w)
        sums |= new_sums
    return len(sums) == (1 << n)

def _greedy_dissociated(order):
    """Greedily build a dissociated set following the given order."""
    current = []
    for x in order:
        test = current + [x]
        if _is_dissociated_fast(test):
            current = test
    return current

def _find_best():
    """Find a large dissociated subset of [1,22]."""
    universe = list(range(1, 23))
    best = []
    
    # Try multiple orderings
    orderings = [
        universe,
        universe[::-1],
        sorted(universe, key=lambda x: x % 5),
        sorted(universe, key=lambda x: x % 7),
        [22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
    ]
    
    # Add random orderings
    random.seed(42)
    for _ in range(200):
        shuffled = universe[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)
    
    for order in orderings:
        result = _greedy_dissociated(order)
        if len(result) > len(best):
            best = result
    
    return best if best else [1]

_cached_result = None

def construct():
    global _cached_result
    if _cached_result is not None:
        return _cached_result
    _cached_result = _find_best()
    return _cached_result
# EVOLVE-BLOCK-END

