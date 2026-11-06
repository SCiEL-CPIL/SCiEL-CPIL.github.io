"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _forms_sunflower(m1, m2, m3):
    # Check if three sets form a sunflower: all three pairwise intersections are equal as sets
    ab = m1 & m2
    ac = m1 & m3
    bc = m2 & m3
    return ab == ac == bc

def _is_compatible(family_masks, new_mask):
    """Check if new_mask can be added to family_masks without forming a sunflower."""
    n = len(family_masks)
    for i in range(n):
        m1 = family_masks[i]
        # Quick check: if m1 and new_mask are identical, they are not distinct, 
        # but the problem says "distinct sets". We ensure distinctness by checking membership.
        # The sunflower definition requires distinct sets A, B, C.
        for j in range(i + 1, n):
            m2 = family_masks[j]
            if _forms_sunflower(m1, m2, new_mask):
                return False
    return True

def _greedy_and_local_add(all_masks, order, deadline):
    family = []
    family_set = set()
    
    # Phase 1: Greedy add in given order
    for idx in order:
        if time.time() >= deadline:
            break
        m = all_masks[idx]
        if m in family_set:
            continue
        if _is_compatible(family, m):
            family.append(m)
            family_set.add(m)
            
    # Phase 2: Local search - try to add any remaining set
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(len(all_masks)):
            if time.time() >= deadline:
                break
            m = all_masks[idx]
            if m in family_set:
                continue
            if _is_compatible(family, m):
                family.append(m)
                family_set.add(m)
                improved = True
                # Reset improved to False and continue loop to check if more can be added
                # Actually, adding one set might enable others? No, adding a set only 
                # creates more constraints. It cannot enable other sets that were previously 
                # blocked by the absence of this set? 
                # Wait. If A and B were blocked because they formed a sunflower with C, 
                # and C is not in the family, they are not blocked by C.
                # The constraint is: No THREE sets in the family form a sunflower.
                # Adding a set D to the family F creates new triples (x, y, D) for x,y in F.
                # It does NOT remove constraints. It only adds them.
                # So if a set S was not compatible with F, it will not become compatible 
                # after adding D to F. 
                # Therefore, the "improved" loop will run at most once effectively? 
                # No, if we scan in a fixed order, we might add S1, then S2 becomes compatible?
                # No, S2 was incompatible with F_old. F_new = F_old U {S1}.
                # If S2 formed a sunflower with x, y in F_old, it still does.
                # If S2 did NOT form a sunflower with any pair in F_old, it WAS compatible.
                # So if we missed it in the first pass (e.g. order issue), we catch it here.
                # But once we start adding, the family grows, making it harder for others.
                # So this loop is basically "find all sets compatible with the current family".
                # It will terminate when no more sets can be added.
                pass
                
    return family

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_masks = []
    for comb in combinations(range(1, 17), 4):
        mask = 0
        for x in comb:
            mask |= (1 << (x - 1))
        all_masks.append(mask)
        
    n_masks = len(all_masks)
    best_family = []
    best_size = 0
    
    # We will perform multiple restarts with different random orders
    # and also try to "shake" the best family found so far
    
    max_restarts = 500 # Will be limited by time
    
    for restart in range(max_restarts):
        if time.time() >= deadline:
            break
            
        # Create a random order
        order = list(range(n_masks))
        random.shuffle(order)
        
        # Run greedy + local add
        family = _greedy_and_local_add(all_masks, order, deadline)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family
            
        # If we have time, try a "shake" on the best family
        # This involves removing a random set from the best family and re-optimizing
        # This is complex to implement correctly within the time limit without risking degradation.
        # Given the small N, random restarts are usually sufficient.
        
    # Convert best_family masks back to sorted lists
    result = []
    for m in best_family:
        s = [i + 1 for i in range(16) if m & (1 << i)]
        result.append(sorted(s))
        
    return result

# EVOLVE-BLOCK-END
