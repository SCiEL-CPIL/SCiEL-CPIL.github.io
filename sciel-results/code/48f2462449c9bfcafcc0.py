"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
from math import lcm
from itertools import combinations

def _is_bad_triple(a, b, c):
    return lcm(a, b) == lcm(a, c) == lcm(b, c)

def _can_add(S, x):
    for a, b in combinations(S, 2):
        if _is_bad_triple(a, b, x):
            return False
    return True

def _greedy(order):
    S = []
    for x in order:
        if _can_add(S, x):
            S.append(x)
    return S

def _fill(S):
    changed = True
    while changed:
        changed = False
        for x in range(1, 129):
            if x not in S and _can_add(S, x):
                S.append(x)
                changed = True
    return S

def construct():
    random.seed(12345)
    best = []
    
    for _ in range(30):
        order = list(range(1, 129))
        random.shuffle(order)
        S = _greedy(order)
        S = _fill(S)
        if len(S) > len(best):
            best = S
    
    S = _greedy(list(range(1, 129)))
    S = _fill(S)
    if len(S) > len(best):
        best = S
    
    return best
# EVOLVE-BLOCK-END
