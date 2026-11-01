import time
import random
import itertools
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    def C(n, r):
        if r > n or r < 0:
            return 0
        if r == 0 or r == n:
            return 1
        r = min(r, n - r)
        result = 1
        for i in range(1, r + 1):
            result = result * (n - i + 1) // i
        return result

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    
    all_k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(all_k_blocks)

    ts_index_map = {ts: i for i, ts in enumerate(all_t_subsets)}

    # Bitmask for each block: bit i is 1 if block covers t-subset i
    # Using Python integers for arbitrary precision bitsets
    block_masks = [0] * num_k_blocks
    # For each t-subset, list of blocks that cover it
    ts_to_blocks = [[] for _ in range(num_t_subsets)]
    
    for j, block in enumerate(all_k_blocks):
        mask = 0
        for ts in itertools.combinations(block, t):
            idx = ts_index_map[ts]
            mask |= (1 << idx)
            ts_to_blocks[idx].append(j)
        block_masks[j] = mask

    # Helper: popcount for Python int
    def popcount(x):
        return bin(x).count('1')

    # Greedy Set Cover
    def greedy_cover():
        uncovered = (1 << num_t_subsets) - 1
        selected = []
        # gains[j] = number of bits in block_masks[j] & uncovered
        gains = [popcount(m) for m in block_masks]
        
        while uncovered:
            if time.time() > deadline - 20.0:
                break
            # Find block with max gain
            max_gain = -1
            best_idx = -1
            # Optimization: Only check blocks that intersect uncovered
            # But checking all is ~75k ops, which is fast.
            for j in range(num_k_blocks):
                if gains[j] > max_gain:
                    max_gain = gains[j]
                    best_idx = j
            
            if best_idx == -1 or max_gain == 0:
                break # Should not happen if uncovered != 0
                
            selected.append(best_idx)
            uncovered &= ~block_masks[best_idx]
            
            # Update gains
            # For each block, new_gain = old_gain - popcount(block_masks[j] & newly_covered)
            # newly_covered = block_masks[best_idx] & (previous_uncovered)
            # But we can just recompute gains for all blocks? 
            # 75k * popcount(183 words) is fast.
            # Let's update incrementally for speed.
            # The bits removed from uncovered are exactly block_masks[best_idx] & previous_uncovered.
            # But we don't have previous_uncovered easily.
            # Let's just recompute gains for all blocks.
            # 75582 * (popcount cost)
            # popcount of 11628-bit int is fast in Python.
            for j in range(num_k_blocks):
                gains[j] = popcount(block_masks[j] & uncovered)
                
        return selected

    def remove_redundant(block_indices):
        if not block_indices:
            return block_indices
        
        # Calculate unique coverage for each block
        # unique_mask[j] = block_masks[j] & ~(OR of all other block_masks)
        # This is O(N^2) bit ops. N~300. 90k ops. Fast.
        
        changed = True
        while changed:
            changed = False
            new_indices = []
            # Precompute OR of all masks
            total_mask = 0
            for idx in block_indices:
                total_mask |= block_masks[idx]
                
            for idx in block_indices:
                # Mask of bits covered by this block that are NOT covered by any other block
                others_mask = total_mask ^ block_masks[idx] # XOR works if we assume disjoint? No.
                # others_mask = total_mask & ~block_masks[idx]
                others_mask = total_mask & ~block_masks[idx]
                unique_mask = block_masks[idx] & ~others_mask
                
                if unique_mask == 0:
                    changed = True
                else:
                    new_indices.append(idx)
            
            block_indices = new_indices
            
            if time.time() > deadline - 5.0:
                break
                
        return block_indices

    def local_search_2for1(block_indices):
        # Try to replace 2 blocks with 1
        # For each pair (i, j), compute unique_i and unique_j
        # If there exists a block k such that (unique_i | unique_j) is subset of block_masks[k],
        # then we can replace i, j with k.
        
        n = len(block_indices)
        if n < 2:
            return block_indices
            
        # Precompute unique masks
        # This is O(N^2)
        unique_masks = {}
        total_mask = 0
        for idx in block_indices:
            total_mask |= block_masks[idx]
            
        for idx in block_indices:
            others_mask = total_mask & ~block_masks[idx]
            unique_masks[idx] = block_masks[idx] & ~others_mask
            
        improved = True
        while improved:
            improved = False
            new_indices = list(block_indices)
            n = len(new_indices)
            
            for i in range(n):
                if time.time() > deadline - 10.0:
                    return new_indices
                for j in range(i + 1, n):
                    idx_i = new_indices[i]
                    idx_j = new_indices[j]
                    
                    needed = unique_masks[idx_i] | unique_masks[idx_j]
                    
                    # If needed is 0, both are redundant, handled by remove_redundant
                    if needed == 0:
                        continue
                        
                    # Find a block k that covers 'needed'
                    # k must not be idx_i or idx_j (if they are in new_indices)
                    # Actually, k can be any block from all_k_blocks
                    # We need k such that (block_masks[k] & needed) == needed
                    
                    # Optimization: Iterate through all blocks? 75k checks per pair.
                    # Pairs ~ 45k. 45k * 75k = 3.3 billion ops. Too slow.
                    
                    # Better: Use ts_to_blocks.
                    # The blocks covering 'needed' are the intersection of ts_to_blocks[s] for all s in needed.
                    # But 'needed' is a bitmask. We need to iterate over the set bits of 'needed'.
                    # Number of bits in 'needed' is small (unique t-subsets of 2 blocks).
                    
                    # Get list of t-subset indices in 'needed'
                    needed_ts = []
                    temp = needed
                    while temp:
                        # Get lowest set bit
                        lsb = temp & -temp
                        idx_ts = lsb.bit_length() - 1
                        needed_ts.append(idx_ts)
                        temp ^= lsb
                        
                    if not needed_ts:
                        continue
                        
                    # Intersect candidate lists
                    candidates = set(ts_to_blocks[needed_ts[0]])
                    for ts_idx in needed_ts[1:]:
                        candidates.intersection_update(ts_to_blocks[ts_idx])
                        if not candidates:
                            break
                    
                    if not candidates:
                        continue
                        
                    # Check if any candidate is not in current solution (or is a different block)
                    # We want to replace idx_i and idx_j with a new block k.
                    # If k is already in the solution (and not i or j), we just remove i and j?
                    # No, if k is in the solution, it's already there.
                    # If k == idx_i or k == idx_j, it's not a swap, it's a removal.
                    # But we established they have unique coverage, so they can't be removed without adding coverage.
                    # So k must be a block NOT in {idx_i, idx_j}.
                    
                    # If k is in new_indices (but not i or j), then adding it is redundant (already there).
                    # We just remove i and j.
                    # If k is NOT in new_indices, we add k and remove i, j.
                    
                    for k in candidates:
                        if k == idx_i or k == idx_j:
                            continue
                            
                        # Check if k is already in new_indices
                        if k in new_indices:
                            # Just remove i and j
                            # This is valid if k covers all unique parts of i and j
                            # Which it does by construction.
                            new_indices.remove(idx_i)
                            new_indices.remove(idx_j)
                            improved = True
                            break
                        else:
                            # Add k, remove i and j
                            # We need to update unique_masks for remaining blocks?
                            # Yes, but we can just restart the loop or update carefully.
                            # For simplicity, let's just do one swap and restart the while loop.
                            new_indices.remove(idx_i)
                            new_indices.remove(idx_j)
                            new_indices.append(k)
                            improved = True
                            break
                            
                    if improved:
                        break
                if improved:
                    break
            
            if improved:
                # Recalculate unique masks for next iteration
                block_indices = new_indices
                total_mask = 0
                for idx in block_indices:
                    total_mask |= block_masks[idx]
                for idx in block_indices:
                    others_mask = total_mask & ~block_masks[idx]
                    unique_masks[idx] = block_masks[idx] & ~others_mask

        return block_indices

    # Main
    # 1. Greedy
    sol_indices = greedy_cover()
    
    # 2. Remove Redundant
    sol_indices = remove_redundant(sol_indices)
    
    # 3. Local Search 2-for-1
    # Run multiple passes as removing blocks changes unique coverage
    for _ in range(10):
        if time.time() > deadline - 5.0:
            break
        new_sol = local_search_2for1(sol_indices)
        if len(new_sol) < len(sol_indices):
            sol_indices = new_sol
            sol_indices = remove_redundant(sol_indices)
        else:
            break
            
    # 4. Final Redundancy Check
    sol_indices = remove_redundant(sol_indices)
    
    # Convert indices back to blocks
    return [list(all_k_blocks[idx]) for idx in sol_indices]

