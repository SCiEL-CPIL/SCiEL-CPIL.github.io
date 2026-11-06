"""Direct constructive approach for minimum maximal no-isosceles set."""
import itertools
import time
import numpy as np
from itertools import combinations

N = 16
DEADLINE = 325.0  # safety margin before 330s kill


def _dist2(a, b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2


def _isosceles(a, b, c):
    d1 = _dist2(a, b)
    d2 = _dist2(a, c)
    d3 = _dist2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _is_maximal(chosen):
    """Check if no grid point can be added to chosen without creating an isosceles triangle."""
    chosen_set = set(chosen)
    for p in itertools.product(range(N), repeat=2):
        if p in chosen_set:
            continue
        # Check if p can be added: for all pairs (a,b) in chosen, (p,a,b) must not be isosceles
        can_add = True
        for a, b in combinations(chosen, 2):
            if _isosceles(p, a, b):
                can_add = False
                break
        if can_add:
            return False  # p can be added, so not maximal
    return True


def _reduce_to_minimal(chosen, start_time, deadline):
    """Try to remove points from chosen while maintaining maximality."""
    chosen = list(chosen)
    improved = True
    while improved:
        if time.time() - start_time > deadline:
            break
        improved = False
        for i in range(len(chosen)):
            if time.time() - start_time > deadline:
                break
            # Try removing chosen[i]
            reduced = chosen[:i] + chosen[i+1:]
            # Check if reduced is still maximal
            if _is_maximal(reduced):
                chosen = reduced
                improved = True
                break  # restart the loop since indices changed
    return chosen


def construct():
    start_time = time.time()
    
    # Greedy maximal construction in row-major order.
    chosen = []
    for p in itertools.product(range(N), repeat=2):
        if time.time() - start_time > DEADLINE:
            break
        if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
            chosen.append(p)
    
    # Reduce to minimal maximal set
    best_set = _reduce_to_minimal(chosen, start_time, DEADLINE)
    
    # Try multiple random restarts with different orderings
    import random
    rng = random.Random(42)
    
    for attempt in range(200):
        if time.time() - start_time > DEADLINE:
            break
        
        # Shuffle the order of points
        points = list(itertools.product(range(N), repeat=2))
        rng.shuffle(points)
        
        chosen = []
        for p in points:
            if time.time() - start_time > DEADLINE:
                break
            if all(not _isosceles(p, a, b) for a, b in itertools.combinations(chosen, 2)):
                chosen.append(p)
        
        # Reduce to minimal maximal set
        reduced = _reduce_to_minimal(chosen, start_time, DEADLINE)
        
        if len(reduced) < len(best_set):
            best_set = reduced[:]
    
    return best_set
# EVOLVE-BLOCK-END
