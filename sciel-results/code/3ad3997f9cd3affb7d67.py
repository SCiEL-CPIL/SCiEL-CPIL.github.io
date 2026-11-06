"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _to_mask(s):
    """Convert a set of elements (1..16) to a 16-bit mask."""
    mask = 0
    for x in s:
        mask |= (1 << (x - 1))
    return mask

def _mask_to_set(mask):
    """Convert a 16-bit mask back to a sorted list of elements."""
    return [i + 1 for i in range(16) if mask & (1 << i)]

def _forms_sunflower_bitmask(new_mask, existing_masks):
    """Check if new_mask forms a sunflower with any two existing masks using bit operations."""
    n = len(existing_masks)
    for i in range(n):
        a = existing_masks[i]
        for j in range(i + 1, n):
            b = existing_masks[j]
            # Sunflower condition: A∩B = A∩C = B∩C
            # i.e., (A & B) == (A & new_mask) == (B & new_mask)
            ab = a & b
            a_new = a & new_mask
            b_new = b & new_mask
            if ab == a_new == b_new:
                return True
    return False

def _greedy_with_order(candidates_masks, deadline):
    """Greedy construction with a given ordering of candidates."""
    family = []
    family_masks = []
    for mask in candidates_masks:
        if time.time() > deadline:
            break
        if not _forms_sunflower_bitmask(mask, family_masks):
            family.append(mask)
            family_masks.append(mask)
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s deadline, 5s safety margin
    
    # Generate all 4-element subsets of [1,16] as bitmasks
    all_masks = [_to_mask(set(c)) for c in combinations(range(1, 17), 4)]
    
    best_family = []
    
    # Try multiple random orderings and greedy constructions
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Lexicographic order (deterministic)
            candidates = list(all_masks)
        else:
            # Random order with seeded randomness for reproducibility
            candidates = list(all_masks)
            random.shuffle(candidates)
        
        family = _greedy_with_order(candidates, deadline)
        
        if len(family) > len(best_family):
            best_family = family[:]
        
        # Early termination if we've hit the maximum
        if len(best_family) >= 250:
            break
    
    # Convert back to sorted lists
    return [sorted(_mask_to_set(m)) for m in best_family]
# EVOLVE-BLOCK-END
