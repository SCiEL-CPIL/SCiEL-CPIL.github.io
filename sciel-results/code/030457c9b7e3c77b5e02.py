"""Rewritten program: greedy + local search for sunflower-free 4-subsets of [16]."""
import time
import random
from itertools import combinations

def _is_sunflower_triple(a, b, c):
    """Check if three sets form a sunflower: all pairwise intersections equal."""
    ab = a & b
    ac = a & c
    bc = b & c
    return ab == ac == bc

def _check_conflict_with_existing(new_set, existing_sets):
    """Check if new_set conflicts with any pair in existing_sets."""
    for i in range(len(existing_sets)):
        for j in range(i + 1, len(existing_sets)):
            if _is_sunflower_triple(new_set, existing_sets[i], existing_sets[j]):
                return True
    return False

def _greedy_construct(order):
    """Greedy construction following given order of candidate sets."""
    chosen = []
    for s in order:
        if not _check_conflict_with_existing(s, chosen):
            chosen.append(s)
    return chosen

def _local_search(family, all_sets, max_iter=5000, deadline=None):
    """Try to improve family by swapping or adding sets."""
    if deadline is not None and time.time() > deadline:
        return family
    
    chosen_set = set(family)
    chosen_list = list(family)
    n = len(chosen_list)
    
    for iteration in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # Try adding a set not in the family
        for s in all_sets:
            if s in chosen_set:
                continue
            # Check if s can be added
            can_add = True
            for i in range(n):
                for j in range(i + 1, n):
                    if _is_sunflower_triple(s, chosen_list[i], chosen_list[j]):
                        can_add = False
                        break
                if not can_add:
                    break
            if can_add:
                chosen_list.append(s)
                chosen_set.add(s)
                n += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try swapping: remove one, add another
        for remove_idx in range(n):
            if deadline is not None and time.time() > deadline:
                return [sorted(s) for s in chosen_set]
            
            removed = chosen_list[remove_idx]
            # Temporarily remove
            chosen_list_no_removed = chosen_list[:remove_idx] + chosen_list[remove_idx+1:]
            n_no_removed = n - 1
            
            for s in all_sets:
                if s in chosen_set and s != removed:
                    continue
                # Check if s can be added to family without removed
                can_add = True
                for i in range(n_no_removed):
                    for j in range(i + 1, n_no_removed):
                        if _is_sunflower_triple(s, chosen_list_no_removed[i], chosen_list_no_removed[j]):
                            can_add = False
                            break
                    if not can_add:
                        break
                if can_add:
                    # Do the swap
                    chosen_list = chosen_list_no_removed + [s]
                    chosen_set.discard(removed)
                    chosen_set.add(s)
                    n = n_no_removed + 1
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            break
    
    return [sorted(s) for s in chosen_set]

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 second deadline, leaving buffer
    
    # Generate all 4-element subsets of [1,16]
    all_sets = [set(c) for c in combinations(range(1, 17), 4)]
    n_total = len(all_sets)
    
    # Precompute a conflict structure: for each pair, find which third sets form sunflowers
    # But this is too large (1820^3), so we'll use on-the-fly checking
    
    # Initial greedy construction with random ordering
    best_family = []
    best_size = 0
    
    # Multiple random restarts
    random.seed(42)
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10:  # Need time for local search
            break
        
        # Random order with some structure
        order = all_sets[:]
        random.shuffle(order)
        
        # Greedy construction
        family = _greedy_construct(order)
        size = len(family)
        
        if size > best_size:
            best_size = size
            best_family = family[:]
        
        # Local search to improve
        if time.time() > deadline - 5:
            break
        
        improved = _local_search(family, all_sets, max_iter=2000, deadline=deadline - 2)
        if len(improved) > best_size:
            best_size = len(improved)
            best_family = improved[:]
    
    # Final local search on best family if time permits
    if time.time() < deadline - 2 and best_family:
        final = _local_search(best_family, all_sets, max_iter=3000, deadline=deadline - 1)
        if len(final) > best_size:
            best_family = final
            best_size = len(final)
    
    return best_family

# EVOLVE-BLOCK-END
