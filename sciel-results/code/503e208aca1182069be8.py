import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds limit
    
    n = 16
    k = 4
    
    # Generate all k-subsets as bitmasks
    all_sets = []
    for combo in combinations(range(n), k):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append(mask)
        
    num_sets = len(all_sets)
    
    # Precompute submasks for each set
    # For a set C, we need to check intersections K which are subsets of C.
    # We only care about K that are possible intersections of two distinct 4-sets.
    # Max intersection size is 3.
    # So K can have size 0, 1, 2, 3.
    submasks = [None] * (1 << n)
    for mask in all_sets:
        subs = []
        # Generate all submasks of mask
        # Since k=4, there are 16 submasks.
        # We can iterate through all subsets of the 4 elements.
        # Or just generate all submasks of the bitmask.
        # Efficient way for small k:
        # Get indices
        indices = []
        temp = mask
        idx = 0
        while temp:
            if temp & 1:
                indices.append(idx)
            temp >>= 1
            idx += 1
            
        for r in range(0, 4): # 0 to 3
            for sub_idx in combinations(indices, r):
                sub_mask = 0
                for x in sub_idx:
                    sub_mask |= (1 << x)
                subs.append(sub_mask)
        submasks[mask] = subs
        
    # Helper to check if a set can be added
    def can_add(c_mask, family):
        # family is a list of masks
        # We need to check if there exist A, B in family such that
        # A & B == A & c_mask == B & c_mask == K
        
        # Group family members by their intersection with c_mask
        # Intersections must be submasks of c_mask
        # We only care about intersections K that are in submasks[c_mask]
        # Note: A & c_mask is always a submask of c_mask.
        
        # Optimization: 
        # If we have two sets A, B in family with same intersection K with c_mask,
        # we must check if A & B == K.
        
        # To avoid O(N^2) pair checks for every K, we can store the first set
        # seen for each K. If we see a second set B with same K, check B & A_first == K.
        # If we see a third set C, check C & A_first == K and C & B == K.
        # This is still potentially O(N^2) if many sets share the same intersection.
        # But with k=4, the number of sets in a family sharing a specific intersection K
        # with a fixed C is limited.
        
        # Let's use a dict: intersection -> list of sets
        # Actually, just storing the list is fine if we check pairs incrementally.
        
        inter_map = {}
        for a in family:
            inter = a & c_mask
            if inter not in inter_map:
                inter_map[inter] = [a]
            else:
                lst = inter_map[inter]
                # Check new set 'a' against all existing in lst
                for b in lst:
                    if (a & b) == inter:
                        return False
                lst.append(a)
        return True

    best_family = []
    best_size = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        random.shuffle(all_sets)
        current_family = []
        
        for c_mask in all_sets:
            if can_add(c_mask, current_family):
                current_family.append(c_mask)
                
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family
            
        # If we reached a very high size, we might break early?
        # Max is 250.
        if best_size >= 250:
            break

    # Convert best_family masks back to sorted lists of integers
    result = []
    for mask in best_family:
        elems = []
        for i in range(n):
            if mask & (1 << i):
                elems.append(i + 1)
        result.append(sorted(elems))
        
    return result
