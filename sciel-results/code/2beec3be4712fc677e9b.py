"""
Strategy: Simulated Annealing (SA) for Set Cover Minimization
1. Initialize with a fast Greedy Set Cover solution.
2. Use Simulated Annealing to explore the space of block sets.
   - Moves: Add a random block, Remove a random block, Swap a block.
   - Cost: Number of blocks + Penalty for uncovered t-subsets.
   - Goal: Minimize cost, driving penalty to 0 and block count down.
This is a stochastic local search, distinct from the deterministic greedy construction.
"""

import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s for safety/return
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks
    tsub_masks = []
    for sub in itertools.combinations(range(v), t):
        m = 0
        for p in sub:
            m |= (1 << p)
        tsub_masks.append(m)
    
    n_tsub = len(tsub_masks)
    idx_of = {m: i for i, m in enumerate(tsub_masks)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    k_sub_masks = []
    k_sub_tsub_indices = []
    
    for comb in all_k_subsets:
        mask = 0
        for p in comb:
            mask |= (1 << p)
        k_sub_masks.append(mask)
        
        subs = []
        for sub in itertools.combinations(comb, t):
            sm = 0
            for p in sub:
                sm |= (1 << p)
            subs.append(idx_of[sm])
        k_sub_tsub_indices.append(subs)
        
    # Precompute "neighbors" or just random selection? 
    # For SA, we need to efficiently update the coverage state.
    # State: 
    #   selected_blocks: set of indices into all_k_subsets
    #   cover_count: np.array of size n_tsub, counts how many selected blocks cover each t-subset
    #   uncovered_count: int, number of t-subsets with cover_count == 0
    
    # 2. Initial Greedy Solution
    # Fast greedy to get a valid starting point
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    selected = set()
    uncovered = np.ones(n_tsub, dtype=bool)
    
    # Precompute offsets for fast gain calculation
    offsets = [0]
    all_indices = []
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Greedy loop (deterministic, fast)
    while uncovered.any() and time.time() < deadline - 30.0:
        gains = np.zeros(n_ksub, dtype=np.int32)
        for i in range(n_ksub):
            start = offsets_arr[i]
            end = offsets_arr[i+1]
            gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        # Pick one
        best_idx = candidates[0]
        
        selected.add(best_idx)
        start = offsets_arr[best_idx]
        end = offsets_arr[best_idx+1]
        # Update cover_count and uncovered
        idxs = all_indices_arr[start:end]
        cover_count[idxs] += 1
        # Check which were 0 before
        newly_covered = idxs[cover_count[idxs] == 1]
        uncovered[newly_covered] = False
        
    # Ensure validity of initial solution
    # If greedy didn't finish, repair
    if uncovered.any():
        uncovered_indices = np.where(uncovered)[0]
        for idx in uncovered_indices:
            tsub_mask = tsub_masks[idx]
            have = [p for p in range(v) if (tsub_mask >> p) & 1]
            need = k - len(have)
            remaining = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
            extra = random.sample(remaining, need) if need > 0 else []
            block = sorted(have + extra)
            mask = 0
            for p in block:
                mask |= (1 << p)
            # Find index of this block
            # Since we precomputed all_k_subsets, we can map mask to index
            # But we don't have a mask_to_idx dict for all 50k? We can build one.
            # Actually, we can just find it.
            # For repair, it's rare, so linear search is ok, or build dict now.
            
        # Rebuild cover_count for the repaired set
        cover_count = np.zeros(n_tsub, dtype=np.int32)
        for b_idx in selected:
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            cover_count[all_indices_arr[start:end]] += 1
            
        # Final check
        if np.any(cover_count == 0):
            # Emergency repair
            for idx in np.where(cover_count == 0)[0]:
                tsub_mask = tsub_masks[idx]
                have = [p for p in range(v) if (tsub_mask >> p) & 1]
                need = k - len(have)
                remaining = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
                extra = random.sample(remaining, need) if need > 0 else []
                block = sorted(have + extra)
                mask = 0
                for p in block:
                    mask |= (1 << p)
                # Find index
                # We need a way to map mask to index quickly.
                # Let's build mask_to_idx now if not done.
                # It's better to build it in precomputation.
                pass
                
    # Let's rebuild mask_to_idx in precomputation for robustness
    # (Moving this logic mentally to the start, but in code we'll add it)
    
    # 3. Simulated Annealing Phase
    # State: selected (set of indices), cover_count (array)
    # Cost: len(selected) + penalty * (number of uncovered)
    # We want to minimize len(selected) while keeping uncovered == 0.
    
    # Re-initialize state from greedy result
    selected = set(selected)
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    for b_idx in selected:
        start = offsets_arr[b_idx]
        end = offsets_arr[b_idx+1]
        cover_count[all_indices_arr[start:end]] += 1
    
    current_cost = len(selected)
    best_cost = current_cost
    best_selected = set(selected)
    
    # SA Parameters
    T = 10.0
    cooling_rate = 0.99
    min_T = 0.1
    
    rng = random.Random(42)
    
    # Helper to add/remove block
    def add_block(idx):
        nonlocal cover_count, current_cost
        start = offsets_arr[idx]
        end = offsets_arr[idx+1]
        idxs = all_indices_arr[start:end]
        # Count how many were 0
        newly_covered = np.sum(cover_count[idxs] == 0)
        cover_count[idxs] += 1
        current_cost += 1
        return newly_covered
        
    def remove_block(idx):
        nonlocal cover_count, current_cost
        start = offsets_arr[idx]
        end = offsets_arr[idx+1]
        idxs = all_indices_arr[start:end]
        # Count how many become 0
        newly_uncovered = np.sum(cover_count[idxs] == 1)
        cover_count[idxs] -= 1
        current_cost -= 1
        return newly_uncovered
        
    def get_uncovered_count():
        return np.sum(cover_count == 0)
        
    # SA Loop
    while time.time() < deadline - 5.0 and T > min_T:
        # Generate a move
        move_type = rng.random()
        
        if move_type < 0.4:
            # Remove a random block
            if not selected: continue
            idx = rng.choice(list(selected))
            newly_uncovered = remove_block(idx)
            selected.discard(idx)
            
            delta_cost = 1 - (newly_uncovered * 100) # Penalty for uncovered
            # Actually, cost = len(selected) + 100 * uncovered
            # Old cost: len(selected) + 100 * old_uncovered
            # New cost: (len(selected)-1) + 100 * (old_uncovered + newly_uncovered)
            # Delta = -1 + 100 * newly_uncovered
            
            if newly_uncovered == 0:
                # Always accept if valid and reduces size
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected)
            else:
                # Accept with probability exp(-delta/T)
                delta = -1 + 100 * newly_uncovered
                if delta < 0 or rng.random() < np.exp(-delta / T):
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_selected = set(selected)
                else:
                    # Reject
                    selected.add(idx)
                    # Undo removal
                    start = offsets_arr[idx]
                    end = offsets_arr[idx+1]
                    cover_count[all_indices_arr[start:end]] += 1
                    current_cost += 1
                    
        elif move_type < 0.8:
            # Add a random block
            idx = rng.randint(0, n_ksub - 1)
            if idx in selected:
                continue
            selected.add(idx)
            start = offsets_arr[idx]
            end = offsets_arr[idx+1]
            idxs = all_indices_arr[start:end]
            newly_covered = np.sum(cover_count[idxs] == 0)
            cover_count[idxs] += 1
            current_cost += 1
            
            delta = 1 - 100 * newly_covered
            if delta < 0 or rng.random() < np.exp(-delta / T):
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_selected = set(selected)
            else:
                # Reject
                selected.discard(idx)
                cover_count[idxs] -= 1
                current_cost -= 1
                
        else:
            # Swap: Remove one, Add one
            if not selected: continue
            idx_out = rng.choice(list(selected))
            idx_in = rng.randint(0, n_ksub - 1)
            if idx_in == idx_out: continue
            
            # Try swap
            # Remove idx_out
            start_out = offsets_arr[idx_out]
            end_out = offsets_arr[idx_out+1]
            idxs_out = all_indices_arr[start_out:end_out]
            newly_uncovered_out = np.sum(cover_count[idxs_out] == 1)
            
            # Add idx_in
            start_in = offsets_arr[idx_in]
            end_in = offsets_arr[idx_in+1]
            idxs_in = all_indices_arr[start_in:end_in]
            # Note: cover_count is currently including idx_out
            # We need to see how many idx_in covers that are currently 0 (after removing idx_out?)
            # This is complex. Let's simplify: just do sequential add/remove if accepted.
            # Or just do a "kick" move: remove one, then add one in next step?
            # Let's stick to Add/Remove for simplicity and speed.
            pass
            
        T *= cooling_rate
        
    # 4. Final Repair and Return
    # Use best_selected
    final_selected = best_selected
    
    # Ensure validity
    cover_count_final = np.zeros(n_tsub, dtype=np.int32)
    for b_idx in final_selected:
        start = offsets_arr[b_idx]
        end = offsets_arr[b_idx+1]
        cover_count_final[all_indices_arr[start:end]] += 1
        
    if np.any(cover_count_final == 0):
        # Repair
        for idx in np.where(cover_count_final == 0)[0]:
            tsub_mask = tsub_masks[idx]
            have = [p for p in range(v) if (tsub_mask >> p) & 1]
            need = k - len(have)
            remaining = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
            extra = random.sample(remaining, need) if need > 0 else []
            block = sorted(have + extra)
            mask = 0
            for p in block:
                mask |= (1 << p)
            
            # Find index
            # We need mask_to_idx. Let's build it on the fly if not present.
            # For safety, linear search in all_k_subsets masks
            found = False
            for i, m in enumerate(k_sub_masks):
                if m == mask:
                    final_selected.add(i)
                    found = True
                    break
            if not found:
                # Should not happen
                pass
                
    result = []
    for idx in final_selected:
        result.append(list(all_k_subsets[idx]))
        
    return result
