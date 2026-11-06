import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Leave 5s margin for safety
    
    # Generate all 4-subsets of [1, 16] as bitmasks
    all_sets = []
    for combo in combinations(range(16), 4):
        mask = 0
        for i in combo:
            mask |= (1 << i)
        all_sets.append(mask)
    
    n_sets = len(all_sets)
    best_family = []
    best_size = 0
    
    # Helper function to check compatibility of a new mask with a list of existing masks
    def is_compatible(new_mask, family):
        len_f = len(family)
        for i in range(len_f):
            a = family[i]
            am = a & new_mask
            for j in range(i):
                b = family[j]
                bm = b & new_mask
                ab = a & b
                if am == bm == ab:
                    return False
        return True

    def greedy_build():
        random.shuffle(all_sets)
        family = []
        for mask in all_sets:
            if time.time() > deadline:
                break
            if is_compatible(mask, family):
                family.append(mask)
        return family

    def local_search_step(family):
        # Returns a new family if an improvement is found, else None
        if not family:
            return None
            
        # Pick a random element to remove
        idx = random.randint(0, len(family) - 1)
        removed_mask = family[idx]
        
        # Create a temporary list without the removed element
        temp_family = family[:idx] + family[idx+1:]
        
        # Create a set of current family masks for fast exclusion
        # Note: removed_mask is NOT in temp_family
        # We want to sample B from all_sets \ set(temp_family)
        # And C from all_sets \ set(temp_family + [B])
        
        # To avoid O(N) filtering and shuffling every time, we can just
        # pick random indices from all_sets and check if they are in temp_family.
        # But temp_family is a list, so check is O(N).
        # Better: maintain a set of masks in the family.
        
        # Let's optimize:
        # We will sample random masks from all_sets.
        # If mask is in temp_family, skip.
        
        family_set = set(temp_family)
        
        # Try to find B
        for _ in range(200): # Sample 200 candidates for B
            b_mask = random.choice(all_sets)
            if b_mask in family_set:
                continue
                
            if not is_compatible(b_mask, temp_family):
                continue
            
            # B is compatible. Try to find C
            temp_family_b = temp_family + [b_mask]
            family_set_b = family_set.union({b_mask})
            
            for _ in range(200): # Sample 200 candidates for C
                c_mask = random.choice(all_sets)
                if c_mask in family_set_b:
                    continue
                
                if is_compatible(c_mask, temp_family_b):
                    # Success: Remove A, Add B, Add C
                    return temp_family_b + [c_mask]
        
        return None

    # Main Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # 1. Greedy Build
        family = greedy_build()
        
        # 2. Local Search Improvement
        # Run LS for a short duration
        ls_start = time.time()
        ls_deadline = min(deadline, ls_start + 5) # Limit LS to 5s per restart
        
        current_family = family
        no_improve_count = 0
        
        while time.time() < ls_deadline:
            new_family = local_search_step(current_family)
            if new_family:
                current_family = new_family
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count > 50:
                    # Stagnant, break LS phase
                    break
        
        if len(current_family) > best_size:
            best_size = len(current_family)
            best_family = current_family[:]
            
            if best_size >= 250:
                break

    # Convert best_family back to list of sorted lists of integers 1..16
    result = []
    for mask in best_family:
        s = []
        for i in range(16):
            if mask & (1 << i):
                s.append(i + 1)
        result.append(sorted(s))
        
    return result
