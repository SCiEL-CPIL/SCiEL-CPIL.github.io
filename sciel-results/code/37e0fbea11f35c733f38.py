import time
import random
import itertools
import numpy as np

def construct(v, k, t):
    # Deadline management: leave 5 seconds for final checks and return
    deadline = time.time() + 115.0
    
    # 1. Precomputation
    # Generate all t-subsets as bitmasks and indices
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
    
    # Precompute t-subset indices for each k-subset
    # k_sub_tsub_indices[i] is a list of t-subset indices covered by block i
    k_sub_tsub_indices = []
    for comb in all_k_subsets:
        indices = []
        for sub in itertools.combinations(comb, t):
            m = 0
            for p in sub:
                m |= (1 << p)
            indices.append(idx_of[m])
        k_sub_tsub_indices.append(indices)
    
    # Convert to numpy arrays for faster access
    all_indices = []
    offsets = [0]
    for indices in k_sub_tsub_indices:
        all_indices.extend(indices)
        offsets.append(len(all_indices))
    
    all_indices_arr = np.array(all_indices, dtype=np.int32)
    offsets_arr = np.array(offsets, dtype=np.int32)
    
    # Mask to index mapping for quick lookup
    mask_to_idx = {}
    for i, comb in enumerate(all_k_subsets):
        m = 0
        for p in comb:
            m |= (1 << p)
        mask_to_idx[m] = i

    # 2. Initial Greedy Construction
    # We use a greedy set cover to get a valid starting point.
    uncovered = np.ones(n_tsub, dtype=bool)
    cover_count = np.zeros(n_tsub, dtype=np.int32)
    blocks_selected = []
    selected_set = set()
    
    # Greedy Loop
    while uncovered.any() and time.time() < deadline - 30.0:
        gains = np.zeros(n_ksub, dtype=np.int32)
        
        # Calculate gains
        unc = uncovered
        ind_arr = all_indices_arr
        off_arr = offsets_arr
        
        for i in range(n_ksub):
            start = off_arr[i]
            end = off_arr[i+1]
            if start < end:
                gains[i] = np.sum(unc[ind_arr[start:end]])
        
        max_gain = np.max(gains)
        
        if max_gain == 0:
            break
        
        # Find all blocks with max gain
        candidates = np.where(gains == max_gain)[0]
        
        # Pick one randomly from candidates to diversify
        best_idx = random.choice(candidates)
        
        # Add block
        blocks_selected.append(best_idx)
        selected_set.add(best_idx)
        
        # Update state
        start = off_arr[best_idx]
        end = off_arr[best_idx+1]
        idxs = ind_arr[start:end]
        
        cover_count[idxs] += 1
        uncovered[idxs] = False
        
    # Ensure validity (repair if time ran out during greedy)
    uncovered_indices = np.where(uncovered)[0]
    for idx in uncovered_indices:
        if time.time() > deadline - 5.0:
            break
        tsub_mask = tsub_masks[idx]
        have = [p for p in range(v) if (tsub_mask >> p) & 1]
        need = k - len(have)
        remaining_points = [p for p in range(v) if not ((tsub_mask >> p) & 1)]
        if need > 0:
            extra = random.sample(remaining_points, need)
        else:
            extra = []
        block = have + extra
        block.sort()
        mask = 0
        for p in block:
            mask |= (1 << p)
        b_idx = mask_to_idx.get(mask)
        if b_idx is not None and b_idx not in selected_set:
            blocks_selected.append(b_idx)
            selected_set.add(b_idx)
            start = offsets_arr[b_idx]
            end = offsets_arr[b_idx+1]
            idxs = all_indices_arr[start:end]
            cover_count[idxs] += 1
            uncovered[idxs] = False

    # 3. Local Search / Simulated Annealing Phase
    # Objective: Minimize len(blocks_selected)
    # Move: Remove a block B, if valid, keep it removed. 
    #       If invalid, try to swap B with a new block B' that covers the "critical" t-subsets lost by removing B.
    
    # To make this efficient, we track "critical" blocks.
    # A block is critical if it is the only block covering some t-subset.
    
    # We will iterate through blocks and try to remove them.
    # If removal fails, we attempt a "repair" by adding a block that covers the deficit, 
    # potentially allowing a swap if the new block is "better" (e.g., covers more critical subsets).
    
    # However, simple "remove if possible" is a form of local search.
    # Let's enhance it:
    # 1. Try to remove any block.
    # 2. If cannot remove, try to replace it with a block that covers the same "essential" t-subsets 
    #    but has a higher "global" coverage or lower redundancy?
    #    Actually, for minimization, we just want to reduce count.
    #    If we can't remove, we are stuck at this count unless we do a multi-move.
    
    # Let's implement a "Random Restart" or "Perturbation" if we get stuck?
    # No, let's stick to a robust local search:
    # 1. Identify all blocks.
    # 2. For each block, check if it can be removed.
    # 3. If yes, remove it.
    # 4. If no, check if it can be "swapped" with a block that allows a subsequent removal?
    #    This is complex.
    
    # Alternative: Use the "Cover Count" to identify flexibility.
    # If a block covers only t-subsets with count > 1, it is redundant and can be removed.
    # This is exactly the "Reduction Phase" in the seed.
    
    # To improve on the seed, the seed does this once (or in a loop until no change).
    # My improvement: Randomized order + Perturbation.
    # 1. Shuffle blocks.
    # 2. Try to remove.
    # 3. If stuck for X iterations, perform a "swap":
    #    - Pick a random block B.
    #    - Pick a random new block B' not in set.
    #    - If (Set - {B} + {B'}) is valid, and maybe has better properties, accept?
    #    - Actually, if valid, the size is the same. We need size reduction.
    #    - So swaps alone don't reduce size.
    
    # The key to beating greedy is often that greedy makes "greedy" choices that prevent later removals.
    # A better strategy:
    # 1. Greedy.
    # 2. Reduction (remove redundant).
    # 3. **Perturbation**: Remove a random block B. This creates "holes" (uncovered t-subsets).
    # 4. **Repair**: Add blocks to cover the holes.
    # 5. **Reduction**: Remove redundant blocks again.
    # 6. If the final size is smaller than before, accept. Else, revert.
    
    current_blocks = blocks_selected[:]
    current_size = len(current_blocks)
    best_blocks = current_blocks[:]
    best_size = current_size
    
    # Precompute for speed
    # We need to quickly check validity and counts.
    
    def get_counts(blocks):
        counts = np.zeros(n_tsub, dtype=np.int32)
        for b in blocks:
            start = offsets_arr[b]
            end = offsets_arr[b+1]
            idxs = all_indices_arr[start:end]
            counts[idxs] += 1
        return counts
    
    # Initial counts
    counts = get_counts(current_blocks)
    
    # Helper: try to reduce counts by removing blocks
    def reduce_counts(counts, blocks):
        # Remove redundant blocks
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            # Shuffle to avoid bias
            random.shuffle(blocks)
            for i in range(len(blocks) - 1, -1, -1):
                b = blocks[i]
                start = offsets_arr[b]
                end = offsets_arr[b+1]
                idxs = all_indices_arr[start:end]
                # If any t-subset has count 1, it's critical
                if np.any(counts[idxs] == 1):
                    continue
                # Redundant
                counts[idxs] -= 1
                blocks.pop(i)
                changed = True
                break # Restart loop
        return blocks, counts

    # Apply initial reduction
    current_blocks, counts = reduce_counts(counts, current_blocks)
    current_size = len(current_blocks)
    if current_size < best_size:
        best_blocks = current_blocks[:]
        best_size = current_size
    
    # Simulated Annealing Loop
    # We try to perturb and repair.
    temp = 1.0
    cooling = 0.95
    
    while time.time() < deadline - 5.0:
        # Perturbation: Remove a random block
        if not current_blocks:
            break
            
        # Choose a block to remove. Prefer blocks that are "less critical" (cover more redundant t-subsets)
        # But for simplicity, just random.
        b_to_remove = random.choice(current_blocks)
        
        # Remove it
        current_blocks.remove(b_to_remove)
        start = offsets_arr[b_to_remove]
        end = offsets_arr[b_to_remove+1]
        idxs = all_indices_arr[start:end]
        counts[idxs] -= 1
        
        # Check for holes
        holes = np.where(counts == 0)[0]
        
        if len(holes) == 0:
            # Valid! We reduced size by 1.
            current_size -= 1
            if current_size < best_size:
                best_blocks = current_blocks[:]
                best_size = current_size
            # Continue trying to remove more
            continue
            
        # We have holes. We must repair.
        # Strategy: Add blocks to cover holes.
        # To minimize added blocks, we cover the "most constrained" holes first.
        # A hole is constrained if few blocks can cover it?
        # Every k-subset containing the t-subset can cover it.
        
        # Simple repair: Greedily cover holes.
        # We need to find blocks that cover the holes.
        # For each hole, find a block that covers it and other holes.
        
        # To speed up, we can just pick a block that covers one hole and add it.
        # But we want to be efficient.
        
        # Let's try to find a block that covers the maximum number of current holes.
        # Since holes are few (only those affected by removal), we can check candidates.
        
        # Candidates: Any k-subset that contains at least one hole's points?
        # Actually, a block covers a hole (t-subset H) if H is a subset of the block.
        
        # Let's iterate through all k-subsets? Too slow.
        # Instead, for each hole, generate candidates?
        
        # Optimization:
        # The holes are t-subsets.
        # We need to add blocks such that every hole is covered.
        
        # Let's use a simple greedy repair on the holes.
        # 1. Identify all t-subsets that are uncovered.
        # 2. Find a block that covers the most uncovered t-subsets.
        # 3. Add it.
        # 4. Repeat.
        
        # To find the best block, we can score blocks by how many holes they cover.
        # But checking all n_ksub blocks is expensive.
        
        # Heuristic:
        # Pick a random hole.
        # Pick a random block that covers it.
        # Add it.
        
        # Let's try to be smarter:
        # For the first hole, pick a block that covers it.
        # To pick a block covering a specific t-subset T:
        # T is a 4-subset. We need to add 3 more points from the remaining 15.
        # There are C(15,3) = 455 such blocks.
        
        # We can sample from these.
        
        repaired = False
        max_repair_steps = 10
        for _ in range(max_repair_steps):
            if time.time() > deadline - 2.0:
                break
                
            holes = np.where(counts == 0)[0]
            if len(holes) == 0:
                repaired = True
                break
                
            # Pick a random hole
            h_idx = random.choice(holes)
            h_mask = tsub_masks[h_idx]
            h_points = [p for p in range(v) if (h_mask >> p) & 1]
            
            # Find a block covering this hole
            # We can precompute a list of blocks covering each t-subset?
            # That's C(19,4) * C(15,3) ~ 3876 * 455 ~ 1.7M entries.
            # Might be too much memory/time to precompute.
            
            # Instead, generate random blocks covering this hole.
            remaining_points = [p for p in range(v) if not ((h_mask >> p) & 1)]
            extra = random.sample(remaining_points, k - t)
            block = h_points + extra
            block.sort()
            mask = 0
            for p in block:
                mask |= (1 << p)
            
            b_idx = mask_to_idx.get(mask)
            
            if b_idx is not None and b_idx not in current_blocks:
                # Add it
                current_blocks.append(b_idx)
                start = offsets_arr[b_idx]
                end = offsets_arr[b_idx+1]
                idxs = all_indices_arr[start:end]
                counts[idxs] += 1
                # Update holes?
                # We just added a block. Counts updated.
        
        if repaired:
            # We have a valid covering again.
            # Size is current_size (which was reduced by 1, then increased by repair steps).
            # Let's check actual size.
            actual_size = len(current_blocks)
            
            # Try to reduce again
            current_blocks, counts = reduce_counts(counts, current_blocks)
            actual_size = len(current_blocks)
            
            if actual_size < best_size:
                best_blocks = current_blocks[:]
                best_size = actual_size
            # Accept this state as the new current state for further perturbations?
            # Yes, if it's valid.
            
        else:
            # Repair failed or timed out. Revert to previous state.
            # This is tricky because we modified counts and blocks in place.
            # We should have copied them.
            # For safety, let's restart the SA loop with a fresh copy if this happens?
            # Or just break?
            # Let's revert to best_blocks for safety and try a different perturbation.
            # But we lost the state.
            # Let's just restart the main loop with best_blocks.
            current_blocks = best_blocks[:]
            counts = get_counts(current_blocks)
            current_size = best_size
            # Reduce again to ensure we are in a minimal state for this set
            current_blocks, counts = reduce_counts(counts, current_blocks)
            current_size = len(current_blocks)
            if current_size < best_size:
                best_blocks = current_blocks[:]
                best_size = current_size

    # Final Reduction on best_blocks
    counts = get_counts(best_blocks)
    best_blocks, counts = reduce_counts(counts, best_blocks)
    
    # Final Conversion
    result = []
    for idx in best_blocks:
        result.append(list(all_k_subsets[idx]))
        
    return result

# EVOLVE-BLOCK-END
