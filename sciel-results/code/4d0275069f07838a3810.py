"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three distinct 4-sets form a sunflower (all three pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _greedy_construct(all_subsets, order, deadline):
    """Greedy construction following a given order of subset indices."""
    chosen = []
    for idx in order:
        if time.time() > deadline:
            break
        s = all_subsets[idx]
        can_add = True
        for i in range(len(chosen)):
            a = all_subsets[chosen[i]]
            for j in range(i + 1, len(chosen)):
                b = all_subsets[chosen[j]]
                if _is_sunflower_triple(a, b, s):
                    can_add = False
                    break
            if not can_add:
                break
        if can_add:
            chosen.append(idx)
    return chosen

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 10s safety margin
    
    # Generate all 4-subsets of [1,16]
    all_subsets = [set(row) for row in combinations(range(1, 17), 4)]
    n = len(all_subsets)
    
    best_family = []
    
    # Multiple greedy passes with different orderings
    orders = []
    
    # Natural order
    orders.append(list(range(n)))
    
    # Random orders with different seeds
    for seed in [42, 123, 456, 789, 1000, 2024, 3141, 2718, 9999, 12345, 54321, 11111]:
        random.seed(seed)
        shuffled = list(range(n))
        random.shuffle(shuffled)
        orders.append(shuffled)
    
    for order in orders:
        if time.time() > deadline:
            break
        family = _greedy_construct(all_subsets, order, deadline)
        if len(family) > len(best_family):
            best_family = family[:]
    
    # Local improvement: try to add sets by removing one conflicting set
    current = best_family[:]
    current_set = set(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for s_idx in range(n):
            if time.time() > deadline:
                break
            if s_idx in current_set:
                continue
            s = all_subsets[s_idx]
            # Try adding directly
            can_add_direct = True
            for i in range(len(current)):
                a = all_subsets[current[i]]
                for j in range(i + 1, len(current)):
                    b = all_subsets[current[j]]
                    if _is_sunflower_triple(a, b, s):
                        can_add_direct = False
                        break
                if not can_add_direct:
                    break
            if can_add_direct:
                current.append(s_idx)
                current_set.add(s_idx)
                improved = True
                break
            else:
                # Try removing one set
                for rem_idx in current:
                    test_can_add = True
                    for i in range(len(current)):
                        if current[i] == rem_idx:
                            continue
                        a = all_subsets[current[i]]
                        for j in range(i + 1, len(current)):
                            if current[j] == rem_idx:
                                continue
                            b = all_subsets[current[j]]
                            if _is_sunflower_triple(a, b, s):
                                test_can_add = False
                                break
                        if not test_can_add:
                            break
                    if test_can_add:
                        current = [x for x in current if x != rem_idx]
                        current.append(s_idx)
                        current_set.discard(rem_idx)
                        current_set.add(s_idx)
                        improved = True
                        break
                if improved:
                    break
    
    result = [sorted(all_subsets[i]) for i in current]
    if len(result) > len(best_family):
        best_family = result
    
    return best_family
# EVOLVE-BLOCK-END
