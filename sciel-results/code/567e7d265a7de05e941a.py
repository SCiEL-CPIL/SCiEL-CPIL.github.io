import random
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 55.0 # 55s limit to be safe
    
    n = 16
    k = 4
    all_masks = []
    for comb in combinations(range(n), k):
        mask = 0
        for x in comb:
            mask |= (1 << x)
        all_masks.append(mask)
    
    num_masks = len(all_masks)
    
    def can_add(family_masks, s_mask):
        # Check all pairs in family
        # Optimization: Early exit
        for i in range(len(family_masks)):
            a = family_masks[i]
            for j in range(i + 1, len(family_masks)):
                b = family_masks[j]
                if (a & b) == (a & s_mask) == (b & s_mask):
                    return False
        return True

    best_size = 0
    best_family_indices = []
    
    # Run multiple restarts
    while time.time() < deadline:
        # 1. Random Greedy
        indices = list(range(num_masks))
        random.shuffle(indices)
        
        family_indices = []
        family_masks = []
        fam_set = set()
        
        for idx in indices:
            mask = all_masks[idx]
            if can_add(family_masks, mask):
                family_indices.append(idx)
                family_masks.append(mask)
                fam_set.add(idx)
                
        current_size = len(family_indices)
        
        if current_size > best_size:
            best_size = current_size
            best_family_indices = family_indices[:]
            if best_size >= 250:
                break
                
        # 2. Local Search
        # Try to add more sets
        # Then try swaps to escape local optima
        local_improvements = 0
        max_local_steps = 200
        
        for _ in range(max_local_steps):
            if time.time() > deadline:
                break
                
            # Try to add a random set
            added = False
            for _ in range(50):
                idx = random.randint(0, num_masks - 1)
                if idx not in fam_set:
                    mask = all_masks[idx]
                    if can_add(family_masks, mask):
                        family_indices.append(idx)
                        family_masks.append(mask)
                        fam_set.add(idx)
                        current_size += 1
                        added = True
                        if current_size > best_size:
                            best_size = current_size
                            best_family_indices = family_indices[:]
                        break
                        
            if added:
                local_improvements += 1
                continue
                
            # If no add, try a swap
            if family_indices:
                rem_pos = random.randint(0, len(family_indices) - 1)
                rem_idx = family_indices[rem_pos]
                rem_mask = family_masks[rem_pos]
                
                # Remove
                family_indices.pop(rem_pos)
                family_masks.pop(rem_pos)
                fam_set.discard(rem_idx)
                
                added_idx = -1
                for _ in range(50):
                    add_idx = random.randint(0, num_masks - 1)
                    if add_idx not in fam_set and add_idx != rem_idx:
                        mask = all_masks[add_idx]
                        if can_add(family_masks, mask):
                            added_idx = add_idx
                            break
                            
                if added_idx != -1:
                    family_indices.append(added_idx)
                    family_masks.append(all_masks[added_idx])
                    fam_set.add(added_idx)
                    # Size unchanged, but we might be able to add more now
                    local_improvements += 1
                else:
                    # Revert
                    family_indices.append(rem_idx)
                    family_masks.append(rem_mask)
                    fam_set.add(rem_idx)

    # Construct result
    if not best_family_indices:
        return []
        
    result = []
    for idx in best_family_indices:
        mask = all_masks[idx]
        subset = [i for i in range(16) if (mask >> i) & 1]
        result.append(sorted(subset))
        
    return result
