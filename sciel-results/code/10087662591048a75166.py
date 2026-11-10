"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

P = {'n': 128}

def _violates(S_set, a, b, c):
    """Check if triple (a,b,c) violates: a+b, a+c, b+c all in S_set."""
    return (a+b) in S_set and (a+c) in S_set and (b+c) in S_set

def _can_add(S, S_set, x):
    """Check if x can be added to S without violation."""
    if x in S_set:
        return False
    for i in range(len(S)):
        for j in range(i+1, len(S)):
            a, b = S[i], S[j]
            if _violates(S_set, x, a, b):
                return False
    return True

def _greedy_build(order):
    """Greedy build following given element order."""
    S = []
    S_set = set()
    for x in order:
        if _can_add(S, S_set, x):
            S.append(x)
            S_set.add(x)
    return S

def construct():
    deadline = time.time() + 55
    n = P['n']
    
    # Strategy 1: Upper half + greedy lower
    S1 = list(range(n//2 + 1, n + 1))
    S1_set = set(S1)
    for x in range(1, n//2 + 1):
        if time.time() > deadline:
            break
        if _can_add(S1, S1_set, x):
            S1.append(x)
            S1_set.add(x)
    
    best = S1
    best_size = len(S1)
    
    # Strategy 2: Greedy from random orders
    for restart in range(50):
        if time.time() > deadline:
            break
        order = list(range(1, n + 1))
        random.shuffle(order)
        S = _greedy_build(order)
        if len(S) > best_size:
            best = S
            best_size = len(S)
    
    # Local search on best: try to add any missing element
    best_set = set(best)
    for x in range(1, n + 1):
        if time.time() > deadline:
            break
        if x not in best_set and _can_add(best, best_set, x):
            best.append(x)
            best_set.add(x)
    
    return best

# EVOLVE-BLOCK-END
