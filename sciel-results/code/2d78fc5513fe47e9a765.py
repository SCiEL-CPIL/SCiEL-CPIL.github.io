import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 58  # 2s safety margin
    
    # 1. Precomputation
    # Generate all 4-element subsets of [1, 16]
    # Elements are 0..15 for bitmask convenience
    all_subsets = list(combinations(range(16), 4))
    N = len(all_subsets) # 1820
    
    # Convert to bitmasks
    masks = np.zeros(N, dtype=np.uint32)
    for i, sub in enumerate(all_subsets):
        m = 0
        for x in sub:
            m |= (1 << x)
        masks[i] = m
    
    # Precompute pairwise intersections as a 2D NumPy array of uint16
    # intersections[i, j] = bitmask of intersection of subset i and subset j
    intersections = np.zeros((N, N), dtype=np.uint16)
    for i in range(N):
        mi = masks[i]
        for j in range(i, N):
            mj = masks[j]
            inter = mi & mj
            intersections[i, j] = inter
            intersections[j, i] = inter

    def check_sunflower(A_idx, B_idx, C_idx):
        """
        Check if subsets A, B, C form a sunflower.
        Condition: A&B == A&C == B&C
        """
        ab = intersections[A_idx, B_idx]
        ac = intersections[A_idx, C_idx]
        if ab != ac:
            return False
        bc = intersections[B_idx, C_idx]
        return ab == bc

    def can_add(C_idx, family_indices):
        """
        Check if C_idx can be added to the family (list of indices) 
        without creating a sunflower with any pair in the family.
        """
        k = len(family_indices)
        for i in range(k):
            A_idx = family_indices[i]
            for j in range(i + 1, k):
                B_idx = family_indices[j]
                if check_sunflower(A_idx, B_idx, C_idx):
                    return False
        return True

    def get_family_size(family_indices):
        return len(family_indices)

    # 2. Greedy Initialization with multiple random restarts
    best_family = []
    best_size = 0
    
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        random.seed(r) # Deterministic per restart for reproducibility in dev, or just random
        # Shuffle indices
        indices = list(range(N))
        random.shuffle(indices)
        
        family = []
        for idx in indices:
            # Check if we can add this
            # Optimization: If family is large, can_add is expensive.
            # But we do this greedily.
            if can_add(idx, family):
                family.append(idx)
        
        if len(family) > best_size:
            best_size = len(family)
            best_family = family[:]
        
        # If we hit the theoretical max or a very high bound, we can stop early?
        # Max possible is 1820, but sunflower-free is much smaller.
        # Let's just keep searching for now.

    # 3. Local Search Improvement
    # Start from the best greedy solution
    family = best_family[:]
    
    # To make swaps efficient, we might want to precompute "conflict partners"
    # But for now, we use the direct check.
    
    # We will try random moves: Add, Remove, Swap
    # Since we want to maximize size, "Add" is best. "Swap" maintains size. "Remove" reduces.
    # We focus on "Add" and "Swap" that allows subsequent "Add"s.
    
    no_improve_count = 0
    max_no_improve = 2000 # If no improvement in 2000 moves, stop or restart?
    
    while time.time() < deadline - 5 and no_improve_count < max_no_improve:
        move_type = random.random()
        
        if move_type < 0.7:
            # Try to ADD a set
            # Pick a random candidate not in family
            # We need an efficient way to pick a candidate.
            # Just pick random indices until we find one not in family.
            # Family size is small compared to N (1820).
            
            # Optimization: Maintain a set of family indices for O(1) check
            # But we are using a list. Let's convert to set for checking membership in the loop.
            # Actually, for picking, we can just try a few random indices.
            
            added = False
            for _ in range(5): # Try 5 random candidates
                cand = random.randint(0, N - 1)
                if cand in family:
                    continue
                if can_add(cand, family):
                    family.append(cand)
                    added = True
                    no_improve_count = 0
                    break
            if not added:
                no_improve_count += 1
                
        elif move_type < 0.9:
            # Try SWAP: Remove one, Add one
            # This helps escape local optima where no single add is possible.
            if len(family) == 0:
                no_improve_count += 1
                continue
            
            # Pick random member to remove
            out_idx_pos = random.randint(0, len(family) - 1)
            out_idx = family[out_idx_pos]
            
            # Pick random candidate to add
            in_idx = -1
            for _ in range(5):
                cand = random.randint(0, N - 1)
                if cand != out_idx and cand not in family:
                    in_idx = cand
                    break
            
            if in_idx == -1:
                no_improve_count += 1
                continue
                
            # Check if family \ {out_idx} + {in_idx} is valid
            # This requires checking if {in_idx} conflicts with pairs in family \ {out_idx}
            # AND if the new family has no other sunflowers (which it shouldn't, as we only changed one element?
            # No, removing out_idx might break existing sunflowers? 
            # The family was valid. Removing an element cannot create a new sunflower.
            # Adding in_idx might create a sunflower with pairs in the remaining family.
            # So we just need to check if in_idx conflicts with any pair in (family \ {out_idx}).
            
            remaining_family = family[:out_idx_pos] + family[out_idx_pos+1:]
            
            if can_add(in_idx, remaining_family):
                family[out_idx_pos] = in_idx
                # Size remains same. We didn't increase size, but we might have unlocked an add.
                # So we shouldn't reset no_improve_count immediately, but maybe decrement it?
                # Let's say it's a neutral move.
                # But often swaps lead to better configurations.
                # Let's treat it as progress if it leads to a valid state (which it does).
                # To avoid infinite loops, we might need a tabu list or just rely on randomness.
                no_improve_count = max(0, no_improve_count - 1)
                
        else:
            # Try REMOVAL (if family is too large and stuck?) 
            # Or just pass.
            # Let's try a "shake": remove 2, add 2?
            # Too complex. Let's just do single removals to allow re-addition.
            if len(family) > 10:
                out_idx_pos = random.randint(0, len(family) - 1)
                family.pop(out_idx_pos)
                no_improve_count = max(0, no_improve_count - 1)

    # Final greedy pass to fill any holes
    # Since the local search might have left valid spots empty
    final_added = True
    while final_added and time.time() < deadline - 1:
        final_added = False
        family_set = set(family)
        for idx in range(N):
            if idx in family_set:
                continue
            if can_add(idx, family):
                family.append(idx)
                final_added = True
                break
        if final_added:
            family_set.add(idx) # Update set for next iter? 
            # Actually, re-iterating from 0 is inefficient.
            # Let's do a single pass.

    # Better final fill:
    family_set = set(family)
    for idx in range(N):
        if idx in family_set:
            continue
        if can_add(idx, family):
            family.append(idx)
            family_set.add(idx)

    # Convert indices back to lists of 1-based integers
    result = []
    for idx in family:
        sub = all_subsets[idx]
        result.append([x + 1 for x in sub])
    
    return result
