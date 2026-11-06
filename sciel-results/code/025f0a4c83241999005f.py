"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

DEADLINE = 80.0  # seconds, safety margin before 90s kill

def _is_sunflower(A, B, C):
    """Three distinct 4-sets form a sunflower iff all three pairwise intersections are equal."""
    return (A & B) == (A & C) == (B & C)

def _can_add(family, s):
    """Check if s can be added to family without creating a sunflower triple."""
    for i in range(len(family)):
        a = family[i]
        for j in range(i + 1, len(family)):
            b = family[j]
            if _is_sunflower(a, b, s):
                return False
    return True

def _greedy_build(order):
    """Build a maximal family by greedily adding sets in the given order."""
    family = []
    for s in order:
        if _can_add(family, s):
            family.append(s)
    return family

def _local_improve(family, all_sets, deadline):
    """Try to improve family by removing one and adding two, or swapping."""
    family = list(family)
    all_set_list = list(all_sets)
    n = len(family)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try single swap: remove one, add one different
        for i in range(n):
            removed = family[i]
            for candidate in all_set_list:
                if candidate == removed:
                    continue
                # Check if candidate can be added to family without removed
                rest = family[:i] + family[i+1:]
                if _can_add(rest, candidate):
                    new_family = rest + [candidate]
                    # This is a valid swap (same size, different composition)
                    # Only accept if it leads to potential for further growth
                    # For now, just track if we found a different valid family
                    # Actually for size maximization, swap keeps same size.
                    # Let's try remove-and-add-two strategy instead
                    break
            else:
                continue
            break
        # Try remove one, add two
        if time.time() >= deadline:
            break
        for i in range(n):
            if time.time() >= deadline:
                break
            removed = family[i]
            rest = family[:i] + family[i+1:]
            # Try to add two sets from candidates not in family
            candidates = [c for c in all_set_list if c != removed and c not in rest]
            added = 0
            new_rest = list(rest)
            for cand in candidates:
                if time.time() >= deadline:
                    break
                if added >= 2:
                    break
                if _can_add(new_rest, cand):
                    new_rest.append(cand)
                    added += 1
            if added >= 2:
                family = new_rest
                n = len(family)
                improved = True
                break
    return family

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    all_4sets = [set(c) for c in combinations(range(1, 17), 4)]
    
    best_family = []
    
    # Multiple greedy restarts with different orderings
    random.seed(42)
    
    # First: natural order greedy
    family = _greedy_build(all_4sets)
    if len(family) > len(best_family):
        best_family = list(family)
    
    # Try several random orderings
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        order = list(all_4sets)
        random.shuffle(order)
        family = _greedy_build(order)
        if len(family) > len(best_family):
            best_family = list(family)
    
    # Local improvement on best family
    if time.time() < deadline - 2:
        best_family = _local_improve(best_family, all_4sets, deadline)
    
    # Final attempt: try to add any remaining set
    if time.time() < deadline - 1:
        family_set = set(frozenset(s) for s in best_family)
        for s in all_4sets:
            if time.time() > deadline - 0.5:
                break
            if frozenset(s) not in family_set:
                if _can_add(best_family, s):
                    best_family.append(s)
                    family_set.add(frozenset(s))
    
    return [sorted(s) for s in best_family]
# EVOLVE-BLOCK-END
