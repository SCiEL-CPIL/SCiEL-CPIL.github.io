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

def _greedy_with_order(sets_in_order, deadline):
    """
    Greedy construction using a given ordering.
    Uses a conflict-tracking structure for speed:
    For each pair of chosen sets, we track the 'core' (intersection) and the
    set of 'forbidden' elements that a third set must avoid to not form a sunflower.
    
    Specifically, if we have chosen sets A and B with intersection I = A & B,
    a new set C forms a sunflower with A and B iff:
      C & A == I  and  C & B == I
    This implies:
      C contains I
      C is disjoint from (A \\ I) and (B \\ I)
      i.e., C is disjoint from (A | B) \\ I
    
    So for each pair (A, B) in the chosen family, we can precompute:
      core = A & B
      forbidden = (A | B) & ~core  (in bitmask terms: (A | B) ^ core, but careful)
      Actually: forbidden = (A | B) - core = (A | B) & ~core
    
    Then C is compatible with pair (A,B) iff:
      (C & core) == core  (C contains core)
      (C & forbidden) == 0  (C is disjoint from forbidden)
    
    We can maintain a list of (core, forbidden) for each pair in the chosen set.
    When adding a new set, we check against all existing pairs.
    """
    chosen = []
    pair_info = []  # list of (core_mask, forbidden_mask)
    
    for mask in sets_in_order:
        if time.time() > deadline:
            break
        
        # Check against all existing pairs
        ok = True
        for core, forbidden in pair_info:
            # C must contain core and be disjoint from forbidden
            if (mask & core) != core:
                ok = False
                break
            if (mask & forbidden) != 0:
                ok = False
                break
        
        if ok:
            # Add the new set
            chosen.append(mask)
            # Update pair_info with new pairs involving this set
            n = len(chosen) - 1
            for i in range(n):
                a = chosen[i]
                b = mask
                core = a & b
                forbidden = (a | b) ^ core  # (a|b) - core
                pair_info.append((core, forbidden))
    
    return chosen

def _local_search_add(family, all_sets, deadline):
    """Try to add more sets to the family by checking all unchosen sets."""
    family = list(family)
    family_set = set(family)
    
    # Build pair_info from current family
    pair_info = []
    n = len(family)
    for i in range(n):
        for j in range(i + 1, n):
            a = family[i]
            b = family[j]
            core = a & b
            forbidden = (a | b) ^ core
            pair_info.append((core, forbidden))
    
    # Try adding each unchosen set
    for mask in all_sets:
        if time.time() > deadline:
            break
        if mask in family_set:
            continue
        
        ok = True
        for core, forbidden in pair_info:
            if (mask & core) != core:
                ok = False
                break
            if (mask & forbidden) != 0:
                ok = False
                break
        
        if ok:
            family.append(mask)
            family_set.add(mask)
            # Update pair_info with new pairs
            n = len(family) - 1
            for i in range(n):
                a = family[i]
                b = mask
                core = a & b
                forbidden = (a | b) ^ core
                pair_info.append((core, forbidden))
    
    return family

def construct():
    """Construct a large sunflower-free family of 4-element subsets of [1,16]."""
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s safety margin
    
    all_sets = _all_4sets_bitmask()
    
    best_family = []
    
    # Try multiple random orderings with the faster greedy
    num_trials = 30
    for trial in range(num_trials):
        if time.time() > deadline - 15:
            break
        
        # Create a random ordering
        order = list(all_sets)
        random.shuffle(order)
        
        # Greedy construction
        family = _greedy_with_order(order, deadline - time.time() - 5)
        
        # Local search to try to add more sets
        if time.time() < deadline - 5:
            family = _local_search_add(family, all_sets, deadline - time.time() - 2)
        
        if len(family) > len(best_family):
            best_family = family
    
    # Convert back to sorted lists
    result = []
    for mask in best_family:
        elements = [i + 1 for i in range(16) if (mask >> i) & 1]
        result.append(sorted(elements))
    
    return result
# EVOLVE-BLOCK-END
