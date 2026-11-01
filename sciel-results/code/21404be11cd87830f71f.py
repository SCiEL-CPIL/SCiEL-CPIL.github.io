import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    # Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subsets_set = {tuple(s): i for i, s in enumerate(t_subsets)}
    num_tsubs = len(t_subsets)
    
    # Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(k_subsets)
    
    # Precompute for each block, which t-subsets it covers as a bitmask
    # For v=19, k=7, t=4: C(19,4) = 3876 t-subsets, which fits in a 4096-bit integer
    block_bitmasks = [0] * num_blocks
    block_to_tsubs = []
    for i, block in enumerate(k_subsets):
        mask = 0
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subsets_set[tsub]
            mask |= (1 << idx)
            covered.append(idx)
        block_bitmasks[i] = mask
        block_to_tsubs.append(covered)
        
    # Precompute for each t-subset, which blocks cover it (for local search)
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for i, tsubs in enumerate(block_to_tsubs):
        for s in tsubs:
            tsub_to_blocks[s].append(i)
            
    # Convert tsub_to_blocks lists to numpy arrays for faster indexing
    tsub_to_blocks_np = [np.array(lst, dtype=np.int32) for lst in tsub_to_blocks]
    
    # Track uncovered t-subsets as a bitmask
    full_mask = (1 << num_tsubs) - 1
    uncovered_mask = full_mask
    
    selected_block_indices = []
    
    # 1. Weighted Greedy Construction
    # Weight each t-subset by 1/scarcity to prioritize covering rare subsets
    # Compute scarcity: number of blocks covering each t-subset
    tsub_scarcity = np.zeros(num_tsubs, dtype=np.int32)
    for s in range(num_tsubs):
        tsub_scarcity[s] = len(tsub_to_blocks[s])
    
    # Weight for greedy: prefer blocks covering scarce t-subsets
    # Use a soft weighting: weight[s] = 1 / (scarcity[s] ** alpha)
    # For simplicity, use bitmask-based greedy but with a twist:
    # We'll do standard max-gain but break ties by preferring blocks that cover scarcer subsets
    while uncovered_mask != 0:
        if time.time() > deadline - 20:  # Leave time for local search
            break
            
        # Find the block with the maximum gain
        best_gain = 0
        best_b = -1
        best_scarcity_sum = 0
        for b in range(num_blocks):
            gain = (block_bitmasks[b] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
                # Compute scarcity sum for tie-breaking
                best_scarcity_sum = 0
                for s in block_to_tsubs[b]:
                    if uncovered_mask & (1 << s):
                        best_scarcity_sum += 1 / max(1, tsub_scarcity[s])
            elif gain == best_gain and gain > 0:
                # Tie-break: prefer block covering scarcer subsets
                scarcity_sum = 0
                for s in block_to_tsubs[b]:
                    if uncovered_mask & (1 << s):
                        scarcity_sum += 1 / max(1, tsub_scarcity[s])
                if scarcity_sum > best_scarcity_sum:
                    best_b = b
                    best_scarcity_sum = scarcity_sum
        if best_b == -1 or best_gain == 0:
            break
            
        selected_block_indices.append(best_b)
        uncovered_mask &= ~block_bitmasks[best_b]
        
    # If we didn't finish greedy (shouldn't happen for v=19), force completion
    while uncovered_mask != 0:
        best_gain = 0
        best_b = -1
        for b in range(num_blocks):
            gain = (block_bitmasks[b] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_b = b
        if best_b == -1:
            break
        selected_block_indices.append(best_b)
        uncovered_mask &= ~block_bitmasks[best_b]
        
    # 2. Local Optimization
    if time.time() < deadline - 5:
        # Maintain cover_count for each t-subset
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
        
        # Helper to check if a block is redundant
        def is_redundant(b_idx, cover_count):
            for s in block_to_tsubs[b_idx]:
                if cover_count[s] == 1:
                    return False
            return True
        
        # Helper to find a block covering all t-subsets in a list
        def find_covering_block(tsubs_to_cover, exclude_set):
            if not tsubs_to_cover:
                return None
            
            first_s = tsubs_to_cover[0]
            candidates = tsub_to_blocks_np[first_s]
            
            valid_candidates = []
            for c in candidates:
                if c in exclude_set:
                    continue
                covers_all = True
                for s in tsubs_to_cover[1:]:
                    if c not in tsub_to_blocks[s]:
                        covers_all = False
                        break
                if covers_all:
                    valid_candidates.append(c)
            
            if not valid_candidates:
                return None
            
            return random.choice(valid_candidates)

        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Step 2a: Redundancy Removal
            redundant_blocks = []
            for b in selected_block_indices:
                if is_redundant(b, cover_count):
                    redundant_blocks.append(b)
            
            if redundant_blocks:
                for b in redundant_blocks:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        cover_count[s] -= 1
                improved = True
                continue
            
            # Step 2b: Try 1-for-1 Swaps (remove one block, add one block that covers all its unique t-subsets)
            unique_counts = {}
            unique_tsubs = {}
            
            for b in selected_block_indices:
                u_count = 0
                u_list = []
                for s in block_to_tsubs[b]:
                    if cover_count[s] == 1:
                        u_count += 1
                        u_list.append(s)
                unique_counts[b] = u_count
                unique_tsubs[b] = u_list
            
            if not unique_counts:
                break
                
            min_unique = min(unique_counts.values())
            candidates_out = [b for b, c in unique_counts.items() if c == min_unique]
            random.shuffle(candidates_out)
            
            swap_made = False
            for b_out in candidates_out:
                s_uniq = unique_tsubs[b_out]
                
                if not s_uniq:
                    # This block is fully redundant, remove it
                    selected_block_indices.remove(b_out)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    swap_made = True
                    improved = True
                    break
                    
                exclude_set = set(selected_block_indices)
                b_in = find_covering_block(s_uniq, exclude_set)
                
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_block_indices.append(b_in)
                    
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] += 1
                    
                    swap_made = True
                    improved = True
                    break
            
            if swap_made:
                continue
            
            # Step 2c: Try 2-for-1 moves (remove two blocks, add one block)
            # This can reduce block count when two blocks' unique t-subsets can be covered by a single new block
            n_selected = len(selected_block_indices)
            if n_selected >= 2:
                # Sample pairs of blocks to try
                pairs_tried = 0
                max_pairs = min(200, n_selected * (n_selected - 1) // 2)
                improved_21 = False
                
                while pairs_tried < max_pairs and time.time() < deadline - 5:
                    i = random.randint(0, n_selected - 1)
                    j = random.randint(0, n_selected - 1)
                    if i == j:
                        continue
                    b1 = selected_block_indices[i]
                    b2 = selected_block_indices[j]
                    pairs_tried += 1
                    
                    # Get unique t-subsets for b1 and b2
                    u1 = [s for s in block_to_tsubs[b1] if cover_count[s] == 1]
                    u2 = [s for s in block_to_tsubs[b2] if cover_count[s] == 1]
                    
                    # If either block has no unique t-subsets, it's redundant (handled above)
                    # For 2-for-1: we need a single block that covers all of u1 ∪ u2
                    all_unique = set(u1) | set(u2)
                    if not all_unique:
                        continue
                    
                    # Find a block that covers all t-subsets in all_unique
                    # and is not already in the solution
                    first_s = list(all_unique)[0]
                    found_b = None
                    for c in tsub_to_blocks_np[first_s]:
                        if c == b1 or c == b2:
                            continue
                        covers_all = True
                        for s in all_unique:
                            if c not in tsub_to_blocks[s]:
                                covers_all = False
                                break
                        if covers_all:
                            found_b = c
                            break
                    
                    if found_b is not None:
                        # Remove b1 and b2, add found_b
                        selected_block_indices.remove(b1)
                        selected_block_indices.remove(b2)
                        selected_block_indices.append(found_b)
                        
                        for s in block_to_tsubs[b1]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[b2]:
                            cover_count[s] -= 1
                        for s in block_to_tsubs[found_b]:
                            cover_count[s] += 1
                        
                        improved_21 = True
                        improved = True
                        break
                
                if improved_21:
                    continue
            
            # Step 2d: Try Add-Minimize (add a block that covers many uncovered t-subsets, then minimize)
            # This helps escape local minima by temporarily increasing then decreasing block count
            if time.time() < deadline - 10:
                # Find t-subsets with lowest coverage (most "scarce")
                # Try adding a block that covers many low-coverage t-subsets
                min_coverage = cover_count.min()
                if min_coverage <= 2:
                    # Find blocks that cover t-subsets with min_coverage
                    low_cov_tsubs = [s for s in range(num_tsubs) if cover_count[s] == min_coverage]
                    if low_cov_tsubs:
                        # Pick a random low-coverage t-subset and find a block covering it
                        s_rand = random.choice(low_cov_tsubs)
                        candidates = tsub_to_blocks_np[s_rand]
                        if len(candidates) > 0:
                            b_add = random.choice(candidates)
                            if b_add not in selected_block_indices:
                                # Add the block
                                selected_block_indices.append(b_add)
                                for s in block_to_tsubs[b_add]:
                                    cover_count[s] += 1
                                
                                # Now try to minimize: remove redundant blocks
                                removed_any = False
                                for b in list(selected_block_indices):
                                    if is_redundant(b, cover_count):
                                        selected_block_indices.remove(b)
                                        for s in block_to_tsubs[b]:
                                            cover_count[s] -= 1
                                        removed_any = True
                                
                                # If we removed at least as many as we added, keep it
                                if removed_any:
                                    improved = True
                                else:
                                    # Revert: remove the added block
                                    selected_block_indices.remove(b_add)
                                    for s in block_to_tsubs[b_add]:
                                        cover_count[s] -= 1
            else:
                break

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    # Deduplicate just in case
    seen = set()
    dedup = []
    for blk in result:
        key = tuple(blk)
        if key not in seen:
            seen.add(key)
            dedup.append(blk)
            
    return dedup
