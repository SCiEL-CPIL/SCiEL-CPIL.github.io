"""Rewritten program: local search to find small maximal no-isosceles sets."""
import time
import random
import itertools
from math import sqrt

N = 16
DEADLINE = 320  # seconds, safety margin

def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2

def _is_isosceles(p, a, b):
    d1 = _dist2(p, a)
    d2 = _dist2(p, b)
    d3 = _dist2(a, b)
    return d1 == d2 or d1 == d3 or d2 == d3

def _can_add(p, S):
    """Check if point p can be added to set S without creating isosceles triangle."""
    for a, b in itertools.combinations(S, 2):
        if _is_isosceles(p, a, b):
            return False
    return True

def _is_maximal(S):
    """Check if S is maximal: no grid point outside S can be added."""
    S_set = set(S)
    for p in itertools.product(range(N), repeat=2):
        if p not in S_set:
            if _can_add(p, S):
                return False
    return True

def _greedy_maximal(order):
    """Build a maximal no-isceles set by greedily adding points in given order."""
    S = []
    S_set = set()
    for p in order:
        if _can_add(p, S):
            S.append(p)
            S_set.add(p)
    return S

def _make_maximal(S):
    """Given a no-isceles set S, greedily add points to make it maximal.
    Returns the maximal set."""
    S = list(S)
    S_set = set(S)
    # Try to add points in a fixed order to make it maximal
    for p in itertools.product(range(N), repeat=2):
        if p not in S_set:
            if _can_add(p, S):
                S.append(p)
                S_set.add(p)
    return S

def _local_search(S, time_limit):
    """Try to reduce the size of maximal set S by removing points."""
    S = list(S)
    S_set = set(S)
    start = time.time()
    
    # Precompute all points
    all_points = list(itertools.product(range(N), repeat=2))
    
    improved = True
    while improved and time.time() - start < time_limit:
        improved = False
        # Try removing each point
        for i in range(len(S) - 1, -1, -1):
            if time.time() - start > time_limit:
                break
            p = S[i]
            # Tentatively remove p
            S_new = S[:i] + S[i+1:]
            # Check if S_new is no-isceles (it is, since subset)
            # Check if S_new is maximal
            if _is_maximal(S_new):
                S = S_new
                S_set = set(S)
                improved = True
                break  # Restart scanning from beginning
    
    return S

def construct():
    start = time.time()
    best = None
    best_size = float('inf')
    
    # Phase 1: Generate multiple maximal sets via randomized greedy
    num_greedy = 50
    for i in range(num_greedy):
        if time.time() - start > DEADLINE:
            break
        order = list(itertools.product(range(N), repeat=2))
        random.shuffle(order)
        S = _greedy_maximal(order)
        if len(S) < best_size:
            best_size = len(S)
            best = S
    
    if best is None:
        # Fallback
        order = list(itertools.product(range(N), repeat=2))
        best = _greedy_maximal(order)
        best_size = len(best)
    
    # Phase 2: Local search on the best set
    if time.time() - start < DEADLINE - 30:
        S = _local_search(best, DEADLINE - 30)
        if len(S) < best_size:
            best_size = len(S)
            best = S
    
    return best
