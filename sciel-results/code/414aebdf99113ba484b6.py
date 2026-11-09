"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower (all pairwise intersections equal)."""
    return (a & b) == (a & c) == (b & c)

def _greedy_construct(order, deadline):
    """Greedy construction given an ordering of all 4-subsets."""
    chosen = []
    chosen_list = []
    for s in order:
        if time.time() > deadline:
            break
        # Check against all pairs in chosen
        conflict = False
        for i in range(len(chosen_list)):
            for j in range(i+1, len(chosen_list)):
                if _is_sunflower_triple(chosen_list[i], chosen_list[j], s):
                    conflict = True
                    break
            if conflict:
                break
        if not conflict:
            chosen_list.append(s)
            chosen.append(s)
    return chosen_list

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    all_sets = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    best_family = []
    
    # First: lexicographic order (baseline)
    if time.time() < deadline:
        family = _greedy_construct(all_sets, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Then: multiple random orderings
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        shuffled = all_sets[:]
        rng.shuffle(shuffled)
        family = _greedy_construct(shuffled, deadline)
        if len(family) > len(best_family):
            best_family = family
            # If we hit the maximum, stop early
            if len(best_family) >= 250:
                break
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
