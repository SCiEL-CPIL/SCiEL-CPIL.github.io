"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _all_4sets_bitmask():
    """Generate all 4-element subsets of [1,16] as bitmasks."""
    sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        sets.append(mask)
    return sets

def _is_sunflower_triple(a, b, c):
    """Check if three bitmask sets form a sunflower (all pairwise intersections equal)."""
    ab = a & b
    ac = a & c
    bc = b & c
    return (ab == ac) and (ac == bc)

def _greedy_with_order(sets_in_order, time_limit):
    """Greedy construction using a given ordering. Returns the family as a list of masks."""
    start_time = time.time()
    chosen = []
    chosen_set = set()
    
    for mask in sets_in_order:
        if time.time() - start_time > time_limit:
            break
        if mask in chosen_set:
            continue
        # Check against all pairs in chosen
        ok = True
        for i in range(len(chosen)):
            a = chosen[i]
            for j in range(i + 1, len(chosen)):
                b = chosen[j]
                if _is_sunflower_triple(a, b, mask):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            chosen.append(mask)
            chosen_set.add(mask)
    
    return chosen

def _local_search(family, all_sets, time_limit):
    """Try to improve the family by adding/removing/swapping sets."""
    start_time = time.time()
    family = list(family)
    family_set = set(family)
    n = len(family)
    
    # Try adding each unchosen set
    for mask in all_sets:
        if time.time() - start_time > time_limit:
            break
        if mask in family_set:
            continue
        # Check if adding this set creates a sunflower
        ok = True
        for i in range(n):
            a = family[i]
            for j in range(i + 1, n):
                b = family[j]
                if _is_sunflower_triple(a, b, mask):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            family.append(mask)
            family_set.add(mask)
            n += 1
    
    return family

def construct():
    """Construct a large sunflower-free family of 4-element subsets of [1,16]."""
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    all_sets = _all_4sets_bitmask()
    n_sets = len(all_sets)  # 1820
    
    best_family = []
    
    # Try multiple random orderings
    num_trials = 20
    for trial in range(num_trials):
        if time.time() > deadline - 10:
            break
        
        # Create a random ordering
        order = list(all_sets)
        random.shuffle(order)
        
        # Greedy construction
        family = _greedy_with_order(order, deadline - time.time() - 5)
        
        # Local search to try to add more sets
        if time.time() < deadline - 5:
            family = _local_search(family, all_sets, deadline - time.time() - 2)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result
# EVOLVE-BLOCK-END
