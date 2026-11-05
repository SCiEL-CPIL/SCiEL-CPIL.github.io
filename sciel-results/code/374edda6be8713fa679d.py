import itertools
import time
import random
import numpy as np
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    all_tsubsets = list(itertools.combinations(range(v), t))
    tsub_to_idx = {s: i for i, s in enumerate(all_tsubsets)}
    n_tsubs = len(all_tsubsets)
    
    all_ksubsets = list(itertools.combinations(range(v), k))
    n_ksubs = len(all_ksubsets)
    
    # block_indices[i] = indices of t-subsets in block i
    # Shape: (n_ksubs, C(k, t))
    t_subsets_per_block = list(itertools.combinations(range(k), t))
    block_indices = np.empty((n_ksubs, len(t_subsets_per_block)), dtype=np.int32)
    
    for i, block in enumerate(all_ksubsets):
        # For each t-subset position in the block, find the global index
        # block is a tuple of k ints
        # We need to map the relative t-subset (e.g., (0,1,2) in block coords) to global values
        # Then look up in tsub_to_idx
        indices = []
        for rel_tsub in t_subsets_per_block:
            global_tsub = tuple(block[j] for j in rel_tsub)
            indices.append(tsub_to_idx[global_tsub])
        block_indices[i] = indices
        
    # Frequency of each t-subset
    freq = np.bincount(block_indices.flatten(), minlength=n_tsubs).astype(np.float64)
    inv_freq = 1.0 / (freq + 1e-9)
    
    # Mapping from t-subset index to list of k-subset indices containing it
    # This is useful for local search (finding replacements)
    tsub_to_ksubs = defaultdict(list)
    for k_idx in range(n_ksubs):
        for t_idx in block_indices[k_idx]:
            tsub_to_ksubs[t_idx].append(k_idx)
            
    best_blocks = None
    best_len = float('inf')
    
    def _greedy_cover_vectorized(mode, seed_val):
        rng = random.Random(seed_val)
        uncovered = np.ones(n_tsubs, dtype=bool)
        selected_indices = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                return None
            
            # Calculate scores
            # Mode 'weighted': sum of inv_freq for uncovered parts
            # Mode 'greedy': count of uncovered parts
            if mode == 'weighted':
                # Weights are inv_freq if uncovered, else 0
                # To avoid creating a new array every time, we can just use 
                # inv_freq where uncovered is true.
                # But block_indices indexing is the bottleneck.
                # Let's stick to simple gain first, it's often robust.
                # Or combine: score = gain + 0.1 * weighted_gain?
                # Let's try pure weighted gain, it handles "hard" triples better.
                weights = np.where(uncovered, inv_freq, 0.0)
                scores = weights[block_indices].sum(axis=1)
            else:
                scores = uncovered[block_indices].sum(axis=1)
                
            max_score = scores.max()
            if max_score == 0:
                return None # Should not happen if valid input
            
            # Find candidates with max score (or close to it)
            # To add diversity, pick from top 10 or top 5%
            threshold = max_score
            candidates = np.where(scores >= threshold)[0]
            
            if len(candidates) > 10:
                # Pick 10 random candidates from the top
                candidates = rng.sample(list(candidates), 10)
            
            best_idx = candidates[rng.randint(0, len(candidates)-1)]
            selected_indices.append(best_idx)
            
            # Update uncovered
            uncovered[block_indices[best_idx]] = False
            
        return [list(all_ksubsets[i]) for i in selected_indices]

    def _remove_redundant(blocks):
        if not blocks:
            return blocks
            
        # Map block to its t-subset indices
        # We need to map the specific block (list of ints) to its global index
        # To avoid hashing lists, we can use a dictionary from tuple to index
        # But we already have all_ksubsets.
        # Let's create a map: tuple(block) -> global index
        ksub_to_global_idx = {ksub: i for i, ksub in enumerate(all_ksubsets)}
        
        n = len(blocks)
        if n == 0: return blocks
        
        # cover_count[t_idx] = number of blocks in 'blocks' that cover t_idx
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        block_tsub_indices = []
        
        for block in blocks:
            global_idx = ksub_to_global_idx[tuple(block)]
            tsubs = block_indices[global_idx]
            block_tsub_indices.append(tsubs)
            for t_idx in tsubs:
                cover_count[t_idx] += 1
                
        improved = list(blocks)
        improved_tsub_indices = list(block_tsub_indices)
        cover_count_local = cover_count.copy()
        
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            for i in range(len(improved) - 1, -1, -1):
                # Check if block i is redundant
                # It is redundant if for all t_idx in improved_tsub_indices[i], cover_count_local[t_idx] > 1
                redundant = True
                for t_idx in improved_tsub_indices[i]:
                    if cover_count_local[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in improved_tsub_indices[i]:
                        cover_count_local[t_idx] -= 1
                    improved.pop(i)
                    improved_tsub_indices.pop(i)
                    changed = True
        return improved

    def _local_search_swap(blocks):
        # Try to swap blocks to enable further redundancy removal
        if not blocks:
            return blocks
            
        ksub_to_global_idx = {ksub: i for i, ksub in enumerate(all_ksubsets)}
        
        # Precompute for current solution:
        # 1. Global indices of blocks in solution
        # 2. Cover counts
        # 3. Unique triples for each block
        
        sol_global_indices = [ksub_to_global_idx[tuple(b)] for b in blocks]
        n = len(blocks)
        
        cover_count = np.zeros(n_tsubs, dtype=np.int32)
        for g_idx in sol_global_indices:
            for t_idx in block_indices[g_idx]:
                cover_count[t_idx] += 1
                
        current_len = n
        
        improved = True
        while improved:
            if time.time() > deadline:
                break
            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break
                # Try to replace block i
                g_idx_i = sol_global_indices[i]
                tsubs_i = block_indices[g_idx_i]
                
                # Find unique triples of block i
                unique_tsubs = []
                for t_idx in tsubs_i:
                    if cover_count[t_idx] == 1:
                        unique_tsubs.append(t_idx)
                        
                if not unique_tsubs:
                    # This block is redundant, should have been removed by _remove_redundant
                    # But maybe it became redundant after a swap?
                    # If so, just remove it.
                    for t_idx in tsubs_i:
                        cover_count[t_idx] -= 1
                    blocks.pop(i)
                    sol_global_indices.pop(i)
                    n -= 1
                    improved = True
                    break # Restart loop as indices changed
                    
                # We need a replacement block j (not in solution) that covers all unique_tsubs
                # And ideally, j covers few other things that are currently uniquely covered by other blocks?
                # No, we just want to maintain validity.
                # The hope is that j covers some triples that are currently covered by multiple blocks,
                # allowing those other blocks to become redundant.
                
                # Candidate j must cover all unique_tsubs.
                # Pick the first unique triple, get candidates covering it.
                first_t = unique_tsubs[0]
                candidates = tsub_to_ksubs[first_t]
                
                found_swap = False
                for j_idx in candidates:
                    if j_idx in sol_global_indices:
                        continue
                    
                    # Check if j covers all other unique_tsubs
                    tsubs_j = block_indices[j_idx]
                    # Convert to set for fast lookup? Or just loop.
                    # unique_tsubs is small (usually 1-3).
                    covers_all = True
                    for t in unique_tsubs:
                        if t not in tsubs_j:
                            covers_all = False
                            break
                            
                    if not covers_all:
                        continue
                        
                    # Potential swap found.
                    # Let's simulate the swap.
                    # Remove i, add j.
                    # Update cover counts.
                    # Check if any block becomes redundant.
                    
                    # To be efficient, we can just perform the swap, run _remove_redundant, 
                    # and see if the length decreases.
                    # This is expensive if done for every candidate.
                    # Limit candidates: only check top 50 candidates for each i?
                    # Or just check the first few.
                    
                    # Let's try the swap.
                    # Backup state
                    old_block = blocks[i]
                    old_g_idx = sol_global_indices[i]
                    
                    # Perform swap
                    for t_idx in tsubs_i:
                        cover_count[t_idx] -= 1
                    for t_idx in tsubs_j:
                        cover_count[t_idx] += 1
                    
                    blocks[i] = list(all_ksubsets[j_idx])
                    sol_global_indices[i] = j_idx
                    
                    # Run redundancy removal on the modified list
                    # We need to pass the list of blocks.
                    # _remove_redundant expects a list of blocks.
                    # It returns a new list.
                    # We need to update cover_count if we keep it.
                    # Actually, _remove_redundant recalculates cover_count.
                    # So we can just call it.
                    
                    new_blocks = _remove_redundant(blocks)
                    
                    if len(new_blocks) < current_len:
                        # Improvement!
                        blocks = new_blocks
                        sol_global_indices = [ksub_to_global_idx[tuple(b)] for b in blocks]
                        n = len(blocks)
                        current_len = n
                        improved = True
                        found_swap = True
                        break # Restart outer loop
                    else:
                        # Revert
                        for t_idx in tsubs_i:
                            cover_count[t_idx] += 1
                        for t_idx in tsubs_j:
                            cover_count[t_idx] -= 1
                        blocks[i] = old_block
                        sol_global_indices[i] = old_g_idx
                        
                if found_swap:
                    break
        return blocks

    # Main Loop
    trial = 0
    while time.time() < deadline - 10: # Leave 10s for final polish
        trial += 1
        
        # Alternate between weighted and greedy
        mode = 'weighted' if trial % 2 == 0 else 'greedy'
        
        blocks = _greedy_cover_vectorized(mode, trial)
        
        if blocks is None:
            continue
            
        # Polish
        blocks = _remove_redundant(blocks)
        blocks = _local_search_swap(blocks)
        blocks = _remove_redundant(blocks) # Final cleanup
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks
            
        if best_len <= 76:
            break

    if best_blocks is None:
        # Fallback
        blocks = _greedy_cover_vectorized('greedy', 0)
        if blocks:
            best_blocks = _remove_redundant(blocks)
        else:
            best_blocks = []

    return best_blocks
