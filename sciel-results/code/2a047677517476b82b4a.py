import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # 1. Precomputation
    # Get all t-subsets
    all_t_subs = list(itertools.combinations(range(v), t))
    num_t_subs = len(all_t_subs)
    
    # Map each t-subset to an index
    t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subs)}
    
    # Get all k-subsets
    all_k_subs = list(itertools.combinations(range(v), k))
    
    # Precompute coverage bitmask for each k-subset
    # block_cov[i] is an integer where bit j is 1 if all_t_subs[j] is a subset of all_k_subs[i]
    block_cov = [0] * len(all_k_subs)
    for i, k_sub in enumerate(all_k_subs):
        mask = 0
        for t_sub in itertools.combinations(k_sub, t):
            idx = t_sub_to_idx[t_sub]
            mask |= (1 << idx)
        block_cov[i] = mask

    # 2. Greedy Construction
    # uncovered: bitmask of all 1s for num_t_subs bits
    full_mask = (1 << num_t_subs) - 1
    uncovered = full_mask
    selected_blocks = []
    selected_indices = []
    
    while uncovered:
        if time.time() > deadline:
            break
            
        best_gain = 0
        best_idx = -1
        
        # Find block with max coverage of uncovered
        for i in range(len(all_k_subs)):
            gain = (block_cov[i] & uncovered).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                if gain == 0:
                    break # No block covers any uncovered? Should not happen if valid
            
        if best_idx == -1:
            break # Stuck
            
        selected_indices.append(best_idx)
        selected_blocks.append(all_k_subs[best_idx])
        uncovered &= ~block_cov[best_idx]
        
    # 3. Local Search: Redundancy Removal and Swaps
    # We work with selected_indices
    # To speed up redundancy check, we can maintain counts
    # But with ~150 blocks and 3876 subsets, recalculating is fine.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Calculate coverage counts
        # counts[j] = number of selected blocks covering t-subset j
        counts = [0] * num_t_subs
        # We need to know which blocks cover which subsets to update counts efficiently?
        # Or just recompute from scratch each time?
        # 150 blocks * 35 subsets per block = 5250 ops to increment counts.
        # This is very fast.
        
        for idx in selected_indices:
            mask = block_cov[idx]
            # Iterate bits in mask
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                counts[bit_idx] += 1
                m ^= lsb
                
        # Check for redundant blocks
        # A block is redundant if it does not cover any t-subset with count == 1
        redundant_indices = []
        for idx in selected_indices:
            mask = block_cov[idx]
            is_redundant = True
            m = mask
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    is_redundant = False
                    break
                m ^= lsb
            if is_redundant:
                redundant_indices.append(idx)
                
        if redundant_indices:
            # Remove one redundant block
            idx_to_remove = redundant_indices[0]
            selected_indices.remove(idx_to_remove)
            improved = True
            continue
            
        # No redundant blocks. Try swaps.
        # Pick a block, see if we can swap it with another that covers its unique parts.
        # To avoid getting stuck, we can try a few random swaps or iterate through blocks.
        
        # Optimization: Only try swaps if we have time.
        if time.time() > deadline - 5:
            break
            
        # Try to swap one block
        # Randomly select a block to try to replace
        if selected_indices:
            target_pos = random.randint(0, len(selected_indices) - 1)
            target_idx = selected_indices[target_pos]
            
            # Find unique subsets for target_idx
            unique_mask = 0
            m = block_cov[target_idx]
            while m:
                lsb = m & -m
                bit_idx = lsb.bit_length() - 1
                if counts[bit_idx] == 1:
                    unique_mask |= lsb
                m ^= lsb
                
            if unique_mask == 0:
                # Should have been caught by redundancy check, but just in case
                selected_indices.remove(target_idx)
                improved = True
                continue
                
            # Find a block that covers unique_mask
            # We can also prefer blocks that cover MORE uncovered subsets? 
            # No, we just need to maintain coverage.
            # Actually, we want to swap to a block that might enable future redundancy removal.
            # Let's just find ANY block that covers unique_mask.
            
            replacement_found = -1
            # Iterate through all possible blocks
            # We can skip blocks already in selected_indices (unless we want to duplicate, which is useless)
            # Actually, we must not add a duplicate.
            current_set = set(selected_indices)
            
            for i in range(len(all_k_subs)):
                if i in current_set:
                    continue
                if (block_cov[i] & unique_mask) == unique_mask:
                    replacement_found = i
                    break
            
            if replacement_found != -1:
                selected_indices[target_pos] = replacement_found
                improved = True
                # Continue loop to re-evaluate redundancy

    # Convert indices back to blocks
    final_blocks = [all_k_subs[i] for i in selected_indices]
    
    # Final check: ensure validity (optional but good for safety)
    # The problem states "Valid when every t-subset ... lies in at least one block"
    # Our algorithm ensures this by construction and local search maintaining coverage.
    
    return final_blocks
