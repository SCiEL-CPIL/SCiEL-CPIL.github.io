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
    return chosen_list

def _local_search(family, deadline):
    """Try to improve family by swapping or adding sets."""
    family_set = set(family)
    family_list = list(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add a set
        for s in all_sets_global:
            if s in family_set:
                continue
            conflict = False
            for i in range(len(family_list)):
                for j in range(i+1, len(family_list)):
                    if _is_sunflower_triple(family_list[i], family_list[j], s):
                        conflict = True
                        break
                if conflict:
                    break
            if not conflict:
                family_list.append(s)
                family_set.add(s)
                improved = True
                break
        # Try to swap: remove one, add another
        if not improved and time.time() < deadline:
            for remove_idx in range(len(family_list)):
                if time.time() > deadline:
                    break
                remove_set = family_list[remove_idx]
                for s in all_sets_global:
                    if s in family_set:
                        continue
                    # Check if adding s and removing remove_set is valid
                    # This is complex; skip for now
                    pass
    return family_list

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    # Precompute all 4-element subsets
    global all_sets_global
    all_sets_global = [set(row) for row in combinations(range(1, 17), 4)]
    n_sets = len(all_sets_global)
    
    best_family = []
    
    # First: lexicographic order (baseline)
    if time.time() < deadline:
        family = _greedy_construct(all_sets_global, deadline)
        if len(family) > len(best_family):
            best_family = family
    
    # Then: multiple random orderings
    rng = random.Random(42)
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        shuffled = all_sets_global[:]
        rng.shuffle(shuffled)
        family = _greedy_construct(shuffled, deadline)
        if len(family) > len(best_family):
            best_family = family
            # If we hit the maximum, stop early
            if len(best_family) >= 250:
                break
    
    # Try local search to improve the best family
    if time.time() < deadline and len(best_family) < 250:
        improved_family = _local_search(best_family, deadline)
        if len(improved_family) > len(best_family):
            best_family = improved_family
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
