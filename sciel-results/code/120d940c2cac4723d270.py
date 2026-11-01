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
    
    # 1. Adaptive Weighted Greedy Construction
    # Point weights that adapt to uncovered t-subsets
    point_weights = np.ones(v, dtype=np.float64)
    
    # Precompute for each block, the set of points it contains (for weight updates)
    block_points = [set(block) for block in k_subsets]
    
    # Precompute for each t-subset, the set of points it contains
    tsub_points = [set(tsub) for tsub in t_subsets]
    
    # Cache for block weight sums
    block_weight_cache = {}
    
    while uncovered_mask != 0:
        if time.time() > deadline - 20:  # Leave time for local search
            break
        
        # Update point weights based on uncovered t-subsets
        # Points in frequently uncovered t-subsets get higher weights
        uncovered_indices = []
        mask = uncovered_mask
        while mask:
            # Get lowest set bit
            lsb = mask & (-mask)
            idx = lsb.bit_length() - 1
            uncovered_indices.append(idx)
            mask ^= lsb
        
        # Update weights: increase weight of points in uncovered t-subsets
        for s in uncovered_indices:
            for p in tsub_points[s]:
                point_weights[p] += 0.1
        
        # Normalize weights to prevent overflow
        max_w = point_weights.max()
        if max_w > 10.0:
            point_weights /= max_w
        
        # Find the block with the maximum weighted gain
        best_score = -1.0
        best_b = -1
        for b in range(num_blocks):
            mask_b = block_bitmasks[b] & uncovered_mask
            if mask_b == 0:
                continue
            gain = mask_b.bit_count()
            # Weighted score: gain * average weight of points in block
            avg_weight = sum(point_weights[p] for p in block_points[b]) / k
            score = gain * avg_weight
            if score > best_score:
                best_score = score
                best_b = b
        
        if best_b == -1:
            # Fallback to unweighted gain
            best_gain = 0
            for b in range(num_blocks):
                gain = (block_bitmasks[b] & uncovered_mask).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_b = b
            if best_b == -1:
                break
        
        selected_block_indices.append(best_b)
        uncovered_mask &= ~block_bitmasks[best_b]
        
        # Decay weights slightly to allow adaptation
        point_weights *= 0.99
        
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
        
    # 2. Local Optimization with adaptive restarts
    if time.time() < deadline - 5:
        # Reset point weights for local search phase
        point_weights = np.ones(v, dtype=np.float64)
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
        
        # Convert selected_block_indices to a set for O(1) membership
        selected_set = set(selected_block_indices)
        
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
        
        # Precompute block membership sets for faster "c in tsub_to_blocks[s]" checks
        # Actually tsub_to_blocks[s] is a list; let's make them sets for O(1) lookup
        tsub_to_blocks_set = [set(lst) for lst in tsub_to_blocks]

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
                    selected_set.discard(b_out)
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] -= 1
                    swap_made = True
                    improved = True
                    break
                    
                b_in = find_covering_block(s_uniq, selected_set)
                
                if b_in is not None:
                    selected_block_indices.remove(b_out)
                    selected_set.discard(b_out)
                    selected_block_indices.append(b_in)
                    selected_set.add(b_in)
                    
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
                max_pairs = min(300, n_selected * (n_selected - 1) // 2)
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
                            if c not in tsub_to_blocks_set[s]:
                                covers_all = False
                                break
                        if covers_all:
                            found_b = c
                            break
                    
                    if found_b is not None:
                        # Remove b1 and b2, add found_b
                        selected_block_indices.remove(b1)
                        selected_set.discard(b1)
                        selected_block_indices.remove(b2)
                        selected_set.discard(b2)
                        selected_block_indices.append(found_b)
                        selected_set.add(found_b)
                        
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
            
            # Step 2d: Adaptive Add-Minimize with point weight guidance
            if time.time() < deadline - 10:
                # Update point weights based on coverage
                for s in range(num_tsubs):
                    if cover_count[s] <= 2:
                        for p in tsub_points[s]:
                            point_weights[p] += 0.05
                
                max_w = point_weights.max()
                if max_w > 5.0:
                    point_weights /= max_w
                
                # Find t-subsets with lowest coverage
                min_coverage = cover_count.min()
                if min_coverage <= 2:
                    # Weighted random selection: prefer blocks with high-weight points
                    low_cov_tsubs = [s for s in range(num_tsubs) if cover_count[s] == min_coverage]
                    if low_cov_tsubs:
                        # Score each candidate block by its point weights
                        scored_candidates = []
                        for s_rand in random.sample(low_cov_tsubs, min(20, len(low_cov_tsubs))):
                            candidates = tsub_to_blocks_np[s_rand]
                            for c in candidates:
                                if c in selected_set:
                                    continue
                                score = sum(point_weights[p] for p in block_points[c]) / k
                                scored_candidates.append((score, c))
                        
                        if scored_candidates:
                            # Pick top candidate
                            scored_candidates.sort(key=lambda x: -x[0])
                            b_add = scored_candidates[0][1]
                            
                            # Add the block
                            selected_block_indices.append(b_add)
                            selected_set.add(b_add)
                            for s in block_to_tsubs[b_add]:
                                cover_count[s] += 1
                            
                            # Now try to minimize: remove redundant blocks
                            removed_any = False
                            for b in list(selected_block_indices):
                                if is_redundant(b, cover_count):
                                    selected_block_indices.remove(b)
                                    selected_set.discard(b)
                                    for s in block_to_tsubs[b]:
                                        cover_count[s] -= 1
                                    removed_any = True
                            
                            # If we removed at least as many as we added, keep it
                            if removed_any:
                                improved = True
                            else:
                                # Revert: remove the added block
                                selected_block_indices.remove(b_add)
                                selected_set.discard(b_add)
                                for s in block_to_tsubs[b_add]:
                                    cover_count[s] -= 1
                
                # Decay weights
                point_weights *= 0.98
            else:
                break

    # Convert selected block indices back to the actual subsets
    result = [list(k_subsets[i]) for i in selected_block_indices]
    
    # Final polish: one more pass of redundancy removal
    if time.time() < deadline - 2:
        final_cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                final_cover_count[s] += 1
        
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for b in list(selected_block_indices):
                redundant = True
                for s in block_to_tsubs[b]:
                    if final_cover_count[s] == 1:
                        redundant = False
                        break
                if redundant:
                    selected_block_indices.remove(b)
                    for s in block_to_tsubs[b]:
                        final_cover_count[s] -= 1
                    changed = True
        
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
