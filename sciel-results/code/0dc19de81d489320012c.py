"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys


def _d2(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _is_iso(a, b, c):
    d1 = _d2(a, b)
    d2 = _d2(a, c)
    d3 = _d2(b, c)
    return d1 == d2 or d1 == d3 or d2 == d3


def _can_add(p, chosen):
    for i in range(len(chosen)):
        for j in range(i + 1, len(chosen)):
            if _is_iso(p, chosen[i], chosen[j]):
                return False
    return True


def _is_maximal(chosen, all_points):
    chosen_set = set(chosen)
    for p in all_points:
        if p in chosen_set:
            continue
        if _can_add(p, chosen):
            return False
    return True


def construct():
    start_time = time.time()
    deadline = start_time + 280  # safety margin
    
    all_points = list(itertools.product(range(N), repeat=2))
    
    # First, do a simple greedy pass to get a baseline
    chosen_greedy = []
    for p in all_points:
        if _can_add(p, chosen_greedy):
            chosen_greedy.append(p)
    
    best = list(chosen_greedy)
    best_size = len(best)
    
    # Try random orderings to find smaller maximal sets
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Shuffle points
        shuffled = list(all_points)
        rng.shuffle(shuffled)
        
        chosen = []
        for p in shuffled:
            if _can_add(p, chosen):
                chosen.append(p)
        
        # Check if maximal
        if _is_maximal(chosen, all_points):
            if len(chosen) < best_size:
                best = chosen
                best_size = len(chosen)
                # If we found a very small set, we could stop, but let's keep going
                if best_size <= 10:
                    break
        
        # If we've been running for a while, try local search on the best set
        if time.time() > start_time + 120 and len(best) > 1:
            # Try removing one point from best and see if it's still maximal
            # Actually, removing a point from a maximal set makes it non-maximal by definition
            # So we need a different approach: start from a partial set and extend greedily
            
            # Alternative: random restart with bias toward early termination
            pass
    
    return best
# EVOLVE-BLOCK-END
