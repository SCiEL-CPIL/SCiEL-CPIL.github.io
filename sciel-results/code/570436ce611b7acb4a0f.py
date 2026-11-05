"""Combined program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 16
    k = 4
    
    # Generate all 4-element subsets as bitmasks
    all_masks = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_masks.append(mask)
    
    m = len(all_masks)
    
    def check_violation(family_masks, new_mask):
        """
        Check if adding new_mask to family_masks creates a sunflower violation.
        For every pair (a, b) in family, check if {a, b, new_mask} has all
        three pairwise intersections equal.
        """
        nf = len(family_masks)
        for i in range(nf):
            a = family_masks[i]
            an = a & new_mask
            for j in range(i + 1, nf):
                b = family_masks[j]
                ab = a & b
                bn = b & new_mask
                if ab == an == bn:
                    return True
        return False
    
    def mask_to_list(mask):
        """Convert bitmask to sorted list of 1..16 elements."""
        result = []
        for i in range(n):
            if mask & (1 << i):
                result.append(i + 1)
        return result
    
    best_size = 0
    best_family = []
    
    # Strategy 1: Greedy in natural order
    family = []
    for mask in all_masks:
        if time.time() > deadline:
            break
        if not check_violation(family, mask):
            family.append(mask)
    
    if len(family) > best_size:
        best_size = len(family)
        best_family = family[:]
    
    # Strategy 2: Multiple random shuffles
    random.seed(42)
    max_attempts = 20
    
    for attempt in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        order = list(range(m))
        random.shuffle(order)
        
        family = []
        for idx in order:
            if time.time() > deadline:
                break
            mask = all_masks[idx]
            if not check_violation(family, mask):
                family.append(mask)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
    
    # Strategy 3: Local improvement - try to add sets that were skipped
    # First, identify which sets are NOT in the best family
    in_family = set(best_family)
    candidates = [mask for mask in all_masks if mask not in in_family]
    
    random.shuffle(candidates)
    
    improved = best_family[:]
    for mask in candidates:
        if time.time() > deadline - 2:
            break
        if not check_violation(improved, mask):
            improved.append(mask)
    
    if len(improved) > best_size:
        best_size = len(improved)
        best_family = improved[:]
    
    # Convert back to sorted lists of 1..16
    result = [mask_to_list(mask) for mask in best_family]
    
    return result
# EVOLVE-BLOCK-END
