"""
Strategy: Vectorized NumPy Greedy with Top-k Randomization and Local Optimization
Uses precomputed incidence structure to perform fast greedy set cover.
Adds local search (block replacement) to escape local minima.
"""

# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    # Leave 10s for final repair and return
    deadline = start_time + 140.0
    
    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_tsub = len(all_t_subsets)
    
    # Map t-subset (tuple) to index
    tsub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_ksub = len(all_k_subsets)
    
    # Precompute for each k-subset, the list of t-subset indices it covers
    # This is a list of lists of integers
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            indices.append(tsub_to_idx[sub])
        k_sub_tsub_indices.append(indices)
    
    # Convert to flat array with offsets for fast numpy access
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # 2. Greedy Construction
    # State: uncovered boolean array
    uncovered = np.ones(n_tsub, dtype=bool)
    
    selected_blocks = []
    used_blocks_set = set()
    
    rng = random.Random(42)
    
    # Helper to calculate gains for all blocks
    def get_gains():
        gains = np.zeros(n_ksub, dtype=np.int32)
        for i in range(n_ksub):
            start = offsets_arr[i]
            end = offsets_arr[i+1]
            if start < end:
                gains[i] = np.sum(uncovered[all_indices_arr[start:end]])
        return gains
    
    # Main greedy loop
    while uncovered.any():
        if time.time() > deadline:
            break
            
        gains = get_gains()
        max_gain = np.max(gains)
        
        if max_gain == 0:
            break
            
        # Get candidates with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # If we have time, try to pick a better candidate among top-k
        # For simplicity, pick randomly from candidates to avoid bias
        # Or pick the one that minimizes future conflicts? 
        # Let's stick to random from max-gain candidates for now.
        # Actually, deterministic first is often safer for reproducibility, 
        # but random helps escape cycles if we had them (we don't in simple greedy).
        # Let's use a small random tie-breaker to potentially find better paths.
        best_idx = candidates[rng.randint(0, len(candidates) - 1)]
        
        # Add block
        selected_blocks.append(best_idx)
        used_blocks_set.add(best_idx)
        
        # Update uncovered
        start = offsets_arr[best_idx]
        end = offsets_arr[best_idx+1]
        uncovered[all_indices_arr[start:end]] = False
        
    # 3. Local Optimization Phase
    # Try to replace blocks to reduce count.
    # If a block B is removed, check if all its t-subsets are still covered by other blocks.
    # This requires maintaining coverage counts.
    
    # Let's re-initialize coverage counts based on selected_blocks
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    for idx in selected_blocks:
        start = offsets_arr[idx]
        end = offsets_arr[idx+1]
        cover_count[all_indices_arr[start:end]] += 1
        
    # We need a way to check if removing a block leaves any t-subset uncovered.
    # A block B can be removed if for all t in B, cover_count[t] > 1.
    
    # We can try to remove blocks in reverse order of addition (LIFO) 
    # or in order of lowest "criticality" (fewest unique t-subsets).
    
    # Let's compute "criticality" for each selected block:
    # number of t-subsets in B that have cover_count == 1.
    # If 0, the block is redundant and can be removed.
    
    # We will iterate and try to remove redundant blocks.
    # Since removing one changes cover counts, we must be careful.
    
    # Simple approach:
    # Loop until no more redundant blocks found.
    changed = True
    while changed and time.time() < deadline - 5.0:
        changed = False
        # Check from the end (last added) first, as they are more likely to be redundant
        for i in range(len(selected_blocks) - 1, -1, -1):
            if time.time() > deadline - 5.0:
                break
                
            idx = selected_blocks[i]
            start = offsets_arr[idx]
            end = offsets_arr[idx+1]
            block_tsubs = all_indices_arr[start:end]
            
            # Check if any t-subset in this block has cover_count == 1
            # If min(cover_count[block_tsubs]) > 1, it's redundant.
            if np.min(cover_count[block_tsubs]) > 1:
                # Remove it
                cover_count[block_tsubs] -= 1
                selected_blocks.pop(i)
                used_blocks_set.discard(idx)
                changed = True
                # Restart the loop from the end? Or continue?
                # Continuing is fine, but indices shift. 
                # Since we are iterating backwards, popping at i is safe for the loop variable 
                # if we decrement i? No, range is fixed.
                # Let's just break and restart the while loop to be safe.
                break
                
    # 4. Final Repair
    # If we exited early or local opt removed too much (it shouldn't, it's safe), 
    # ensure all t-subsets are covered.
    uncovered = np.zeros(n_tsub, dtype=bool)
    # Re-calculate uncovered from scratch to be safe? 
    # No, cover_count is maintained.
    # uncovered = (cover_count == 0)
    
    uncovered_indices = np.where(cover_count == 0)[0]
    
    if len(uncovered_indices) > 0:
        # Need to add blocks to cover these
        # Greedy repair
        for idx in uncovered_indices:
            if time.time() > deadline - 2.0:
                break
                
            tsub = all_t_subsets[idx]
            # Find a k-subset containing this t-subset
            # We can just pick one arbitrarily, e.g., the first one in all_k_subsets that contains it
            # Or generate one.
            # Since we have all_k_subsets, we can check.
            # But checking all is slow.
            # Let's construct one.
            tset = set(tsub)
            remaining = [p for p in range(v) if p not in tset]
            need = k - t
            if need > 0:
                extra = rng.sample(remaining, need)
                block = list(tsub) + extra
                block.sort()
            else:
                block = list(tsub)
                
            # Find index in all_k_subsets
            # We need a map from tuple to index for k-subsets
            # We didn't create one. Let's create a set for quick lookup?
            # Or just use a linear search? C(19,7) is 50k, so linear search is ~50k ops.
            # Repair might add many blocks.
            # Let's create a dict for k-subsets.
            # Wait, we can just use the existing all_k_subsets list and a dict.
            
            if not hasattr(construct, 'k_sub_map'):
                construct.k_sub_map = {comb: i for i, comb in enumerate(all_k_subsets)}
                
            if tuple(block) in construct.k_sub_map:
                k_idx = construct.k_sub_map[tuple(block)]
                if k_idx not in used_blocks_set:
                    selected_blocks.append(k_idx)
                    used_blocks_set.add(k_idx)
                    # Update cover count
                    start = offsets_arr[k_idx]
                    end = offsets_arr[k_idx+1]
                    cover_count[all_indices_arr[start:end]] += 1

    # 5. Convert to result
    result = []
    for idx in selected_blocks:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
