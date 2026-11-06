"""Rewritten: greedy with efficient constraint checking."""
import time
from itertools import combinations

N = 192
K = 3
MAX_CARD = 48
DEADLINE_MARGIN = 5.0

def _can_add(a, S):
    """Check if a can be added to S without violating the constraint."""
    # Condition 1: a does not divide sum of any 3 distinct elements in S
    for b, c, d in combinations(S, 3):
        if (b + c + d) % a == 0:
            return False
    # Condition 2: for each b in S, b does not divide (a + c + d) for any c,d in S\{b}
    for b in S:
        others = [x for x in S if x != b]
        for c, d in combinations(others, 2):
            if (a + c + d) % b == 0:
                return False
    return True

def _greedy(order, deadline):
    S = []
    for a in order:
        if time.time() > deadline:
            break
        if _can_add(a, S):
            S.append(a)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + (90.0 - DEADLINE_MARGIN)
    
    best = []
    
    # Multiple orders
    orders = [
        list(range(N, 0, -1)),
        list(range(1, N + 1)),
        [x for x in range(2, N + 1, 2)] + [x for x in range(1, N + 1, 2)],
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        result = _greedy(order, deadline)
        if len(result) > len(best):
            best = result
    
    # Local improvement: try to add more elements
    S = list(best)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in range(1, N + 1):
            if time.time() > deadline:
                break
            if a in S:
                continue
            if _can_add(a, S):
                S.append(a)
                improved = True
                break
    
    return S[:MAX_CARD] if len(S) > MAX_CARD else S
