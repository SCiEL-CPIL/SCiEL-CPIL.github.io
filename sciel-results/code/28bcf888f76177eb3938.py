import time
import random
from itertools import combinations
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin
    
    # 1. Precomputation
    # Generate all 4-subsets of [1, 16]
    all_sets = [set(s) for s in combinations(range(1, 17), 4)]
    n_sets = len(all_sets)
    
    # Convert to bitmasks for fast intersection
    all_bitmasks = []
    for s in all_sets:
        mask = 0
        for x in s:
            mask |= (1 << (x - 1))
        all_bitmasks.append(mask)
    
    # Helper: Check if candidate mask can be added to a list of family masks
    def is_valid_addition(cand_mask, family_masks):
        # Group family members by their intersection with cand
        # Key: intersection bitmask, Value: list of family masks with that intersection
        groups = defaultdict(list)
        for fm in family_masks:
            inter = cand_mask & fm
            groups[inter].append(fm)
        
        # For each group, check if any pair has intersection equal to the group key
        for S, members in groups.items():
            if len(members) < 2:
                continue
            n = len(members)
            # Check all pairs in this group
            for i in range(n):
                mi = members[i]
                for j in range(i + 1, n):
                    if (mi & members[j]) == S:
                        return False
        return True

    # Helper: Greedy construction given an order
    def greedy_construct(order, family_indices):
        family_masks = []
        for idx in order:
            if is_valid_addition(all_bitmasks[idx], family_masks):
                family_masks.append(all_bitmasks[idx])
                family_indices.append(idx)
        return family_masks, family_indices

    rng = random.Random(42)
    best_family_indices = []
    best_size = 0
    
    # Phase 1: Multiple Greedy Initializations
    # Try a few different orderings to get a good starting point
    # 1. Natural order
    # 2. Reversed order
    # 3. Random shuffles
    
    init_orders = []
    init_orders.append(list(range(n_sets)))
    init_orders.append(list(range(n_sets - 1, -1, -1)))
    for _ in range(5):
        o = list(range(n_sets))
        rng.shuffle(o)
        init_orders.append(o)
        
    for order in init_orders:
        if time.time() > deadline - 10:
            break
        fam_masks, fam_indices = greedy_construct(order, [])
        if len(fam_indices) > best_size:
            best_size = len(fam_indices)
            best_family_indices = list(fam_indices)
            
    # Phase 2: Local Search (Hill Climbing with Swaps)
    # We maintain the best family found so far and try to improve it.
    # We also keep a few "current" states if we want to do restarts, 
    # but for simplicity and speed, we'll refine the best one found in Phase 1.
    
    # To be more robust, we can run local search on the top few initializations.
    # But let's focus on the single best one first, and if time permits, restart.
    
    def local_search(family_indices, family_masks, max_time_left=5.0):
        """
        Perform local search on the given family.
        Returns the improved family indices and masks.
        """
        current_size = len(family_indices)
        in_family = [False] * n_sets
        for idx in family_indices:
            in_family[idx] = True
            
        # Pre-compute a list of indices not in family for faster random sampling?
        # Or just sample and check.
        
        # We will use a while loop until time runs out
        # Strategy:
        # 1. Try to add a random set.
        # 2. If fail, try a swap: remove one, add one (or two).
        
        while time.time() < deadline - max_time_left:
            # 1. Try to add
            # Sample a few random candidates
            added = False
            for _ in range(10): # Try 10 random additions
                cand_idx = rng.randint(0, n_sets - 1)
                if in_family[cand_idx]:
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], family_masks):
                    family_masks.append(all_bitmasks[cand_idx])
                    family_indices.append(cand_idx)
                    in_family[cand_idx] = True
                    current_size += 1
                    added = True
                    break
            
            if added:
                continue
            
            # 2. Try Swap: Remove 1, Add 1 or 2
            # Pick a random set to remove
            rem_pos = rng.randint(0, len(family_indices) - 1)
            rem_idx = family_indices[rem_pos]
            rem_mask = family_masks[rem_pos]
            
            # Temporarily remove
            family_indices[rem_pos] = -1
            family_masks[rem_pos] = None
            in_family[rem_idx] = False
            
            # We have a "hole" in the list. 
            # For simplicity in `is_valid_addition`, we pass a filtered list or handle None.
            # `is_valid_addition` iterates over `family_masks`. If we have Nones, it will crash on `&`.
            # So we must filter or swap the last element into the hole.
            
            # Let's swap the last element into the hole to keep the list dense.
            last_pos = len(family_indices) - 1
            if rem_pos != last_pos:
                family_indices[rem_pos] = family_indices[last_pos]
                family_masks[rem_pos] = family_masks[last_pos]
                in_family[family_indices[rem_pos]] = True # Already True, just ensuring logic is clear
            # Now pop the last element
            last_idx = family_indices.pop()
            last_mask = family_masks.pop()
            in_family[last_idx] = False # It was removed from the list, but logically it's still in the family? 
            # Wait, the logic of `in_family` tracks membership.
            # If we swap `last` to `rem_pos`, `last` is still in family.
            # We removed `rem_idx`.
            # So `in_family[rem_idx]` should be False.
            # `in_family[last_idx]` should remain True.
            # The list now contains `last_idx` at `rem_pos`.
            # We popped the last slot.
            # So the family is now missing `rem_idx`.
            
            # Try to add 1 or 2 sets to fill the gap or exceed it.
            added_count = 0
            # Try adding 2 sets
            for attempt in range(2):
                # Try to add a random set
                cand_idx = rng.randint(0, n_sets - 1)
                if in_family[cand_idx]:
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], family_masks):
                    family_masks.append(all_bitmasks[cand_idx])
                    family_indices.append(cand_idx)
                    in_family[cand_idx] = True
                    added_count += 1
                    # If we added 2, we are ahead by 1. Stop.
                    if added_count == 2:
                        break
            
            # If we added 0 or 1, we didn't improve size (or kept same if 1).
            # If we added 1, size is same. If 0, size decreased.
            # If size decreased, we MUST revert to keep the family valid and large.
            # If size same, we can accept the swap (different structure) or revert.
            # Let's accept if size >= previous size (i.e., added_count >= 1).
            # But wait, if added_count == 0, we are smaller. Revert.
            # If added_count == 1, we are same size. Accept (exploration).
            # If added_count == 2, we are larger. Accept.
            
            if added_count == 0:
                # Revert: remove the added ones (none), restore removed one
                family_masks.append(rem_mask)
                family_indices.append(rem_idx)
                in_family[rem_idx] = True
            # If added_count >= 1, we keep the new state.
            # Note: if added_count == 1, the set `rem_idx` is gone, and `cand` is in.
            # The list length is correct.
            
            # Update best
            if len(family_indices) > best_size:
                best_size = len(family_indices)
                best_family_indices = list(family_indices)
                
        return family_indices, family_masks

    # Run local search on the best initial solution
    if best_family_indices:
        fam_masks = [all_bitmasks[i] for i in best_family_indices]
        fam_indices = list(best_family_indices)
        
        # Run local search for a significant chunk of time
        # We pass a dynamic time limit
        local_search(fam_indices, fam_masks, max_time_left=3.0)
        
        # If we have time, we can try to restart from scratch with a different seed 
        # and run local search again, keeping the global best.
        # But local search is the main driver.
        
        # Let's do a second pass of local search on the current best, 
        # perhaps with a different random seed or strategy?
        # The function `local_search` modifies the lists in place.
        # We should probably copy the best and run again? 
        # Or just continue running on the current state.
        
        # Let's just continue running on the current state if time remains.
        # The `while` loop in `local_search` checks `deadline`.
        # So calling it again will just continue where it left off?
        # No, it has a `max_time_left` parameter which sets the internal deadline.
        # We should just let it run until the global deadline.
        
        # Actually, the `local_search` function above has `while time.time() < deadline - max_time_left`.
        # If we call it once with `max_time_left=3.0`, it stops 3s before end.
        # We can call it again with `max_time_left=1.0` to use the last 3s?
        # No, it uses the global `deadline`.
        
        # Let's refine: just run the loop until `deadline - 1.0`.
        
        # Re-implementing the loop to use full time
        current_fam_indices = list(best_family_indices)
        current_fam_masks = [all_bitmasks[i] for i in current_fam_indices]
        in_family = [False] * n_sets
        for idx in current_fam_indices:
            in_family[idx] = True
        
        while time.time() < deadline - 1.0:
            # 1. Try to add
            added = False
            for _ in range(20):
                cand_idx = rng.randint(0, n_sets - 1)
                if in_family[cand_idx]:
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], current_fam_masks):
                    current_fam_masks.append(all_bitmasks[cand_idx])
                    current_fam_indices.append(cand_idx)
                    in_family[cand_idx] = True
                    added = True
                    break
            
            if added:
                if len(current_fam_indices) > best_size:
                    best_size = len(current_fam_indices)
                    best_family_indices = list(current_fam_indices)
                continue
            
            # 2. Swap
            if not current_fam_indices:
                break
                
            rem_pos = rng.randint(0, len(current_fam_indices) - 1)
            rem_idx = current_fam_indices[rem_pos]
            rem_mask = current_fam_masks[rem_pos]
            
            # Remove rem_idx
            last_pos = len(current_fam_indices) - 1
            if rem_pos != last_pos:
                current_fam_indices[rem_pos] = current_fam_indices[last_pos]
                current_fam_masks[rem_pos] = current_fam_masks[last_pos]
            current_fam_indices.pop()
            current_fam_masks.pop()
            in_family[rem_idx] = False
            
            # Try to add
            added_count = 0
            for _ in range(2):
                cand_idx = rng.randint(0, n_sets - 1)
                if in_family[cand_idx]:
                    continue
                if is_valid_addition(all_bitmasks[cand_idx], current_fam_masks):
                    current_fam_masks.append(all_bitmasks[cand_idx])
                    current_fam_indices.append(cand_idx)
                    in_family[cand_idx] = True
                    added_count += 1
                    if added_count == 2:
                        break
            
            if added_count == 0:
                # Revert
                current_fam_masks.append(rem_mask)
                current_fam_indices.append(rem_idx)
                in_family[rem_idx] = True
            else:
                # Accept
                if len(current_fam_indices) > best_size:
                    best_size = len(current_fam_indices)
                    best_family_indices = list(current_fam_indices)

    # Final return
    return [sorted(all_sets[i]) for i in best_family_indices]

