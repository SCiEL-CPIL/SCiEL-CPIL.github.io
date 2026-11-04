import itertools
import time

def construct(v, k, t):
    deadline = time.time() + 115  # Safety margin
    
    # 1. Precompute all t-subsets and map to index
    all_t_subs = list(itertools.combinations(range(v), t))
    n_t_subs = len(all_t_subs)
    sub_to_idx = {s: i for i, s in enumerate(all_t_subs)}
    
    # 2. Precompute all k-blocks
    all_k_blocks = list(itertools.combinations(range(v), k))
    n_k_blocks = len(all_k_blocks)
    
    # 3. Precompute coverage masks for blocks and reverse index for subsets
    block_masks = [0] * n_k_blocks
    # blocks_containing_subset[i] is a list of block indices that contain t-subset i
    blocks_containing_subset = [[] for _ in range(n_t_subs)]
    
    for i, block in enumerate(all_k_blocks):
        mask = 0
        for t_sub in itertools.combinations(block, t):
            idx = sub_to_idx[t_sub]
            mask |= (1 << idx)
            blocks_containing_subset[idx].append(i)
        block_masks[i] = mask
        
    # 4. Greedy Construction
    uncovered_mask = (1 << n_t_subs) - 1
    selected_block_indices = []
    
    while uncovered_mask != 0:
        if time.time() > deadline - 20:
            break
            
        best_idx = -1
        best_gain = -1
        
        for i in range(n_k_blocks):
            gain = (block_masks[i] & uncovered_mask).bit_count()
            if gain > best_gain:
                best_gain = gain
                best_idx = i
                # Optional: break if gain == n_t_subs (unlikely)
        
        if best_idx == -1 or best_gain == 0:
            break
            
        selected_block_indices.append(best_idx)
        uncovered_mask &= ~block_masks[best_idx]
        
    # 5. Local Search
    current_blocks = set(selected_block_indices)
    
    # Helper to get unique subsets for a block given cover counts
    # cover_count is a list of counts for each t-subset
    
    def get_unique_subsets(block_idx, cover_count):
        mask = block_masks[block_idx]
        uniques = []
        m = mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            if cover_count[idx] == 1:
                uniques.append(idx)
            m ^= lsb
        return uniques

    # Main local search loop
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # Compute cover counts
        cover_count = [0] * n_t_subs
        for b_idx in current_blocks:
            m = block_masks[b_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                cover_count[idx] += 1
                m ^= lsb
                
        # Pass 1: Redundancy Removal
        blocks_to_remove = []
        for b_idx in list(current_blocks):
            uniques = get_unique_subsets(b_idx, cover_count)
            if not uniques:
                blocks_to_remove.append(b_idx)
                
        if blocks_to_remove:
            for b_idx in blocks_to_remove:
                current_blocks.remove(b_idx)
            improved = True
            continue # Restart loop to recalculate counts
            
        # Pass 2: 2-to-1 or 2-to-0 Merge
        # We look for pairs (i, j) such that their unique subsets can be covered by one block C
        # or are already covered by other blocks (if C is in set)
        
        # To make this efficient, we only check pairs where both have unique subsets?
        # If one has no unique subsets, it would have been removed in Pass 1.
        # So all blocks in current_blocks have at least 1 unique subset.
        
        blocks_list = list(current_blocks)
        n_blocks = len(blocks_list)
        
        found_merge = False
        for i_idx in range(n_blocks):
            if time.time() > deadline - 10: break
            b_i = blocks_list[i_idx]
            uniques_i = get_unique_subsets(b_i, cover_count)
            
            for j_idx in range(i_idx + 1, n_blocks):
                b_j = blocks_list[j_idx]
                uniques_j = get_unique_subsets(b_j, cover_count)
                
                # Union of unique subsets
                # We need to check if there is a block C covering uniques_i U uniques_j
                # Represent uniques as a set of indices
                # If union is empty, we remove both (handled in Pass 1, but just in case)
                
                # To find C, we intersect the lists of blocks containing each subset in the union
                # If the union is large, this is slow. But uniques are usually small.
                
                # Optimization: If uniques_i and uniques_j are disjoint, the union size is sum.
                # We need a block C that contains ALL of them.
                
                # Start with the list of blocks containing the first subset in the union
                # Then intersect with lists of subsequent subsets.
                
                union_uniques = set(uniques_i) | set(uniques_j)
                
                if not union_uniques:
                    # Should not happen if Pass 1 worked, but if it does, remove both
                    current_blocks.remove(b_i)
                    current_blocks.remove(b_j)
                    improved = True
                    found_merge = True
                    break
                    
                # Sort by list length to minimize intersection cost
                sorted_uniques = sorted(union_uniques, key=lambda x: len(blocks_containing_subset[x]))
                
                candidate_blocks = set(blocks_containing_subset[sorted_uniques[0]])
                
                for s_idx in sorted_uniques[1:]:
                    if not candidate_blocks:
                        break
                    candidate_blocks &= set(blocks_containing_subset[s_idx])
                    
                if candidate_blocks:
                    # We found at least one block C that covers all unique subsets of i and j
                    # We can replace i and j with C
                    # If C is already in current_blocks, we just remove i and j
                    # If C is not in current_blocks, we remove i and j and add C
                    
                    # Pick the first candidate
                    c_block = next(iter(candidate_blocks))
                    
                    if c_block in current_blocks:
                        current_blocks.remove(b_i)
                        current_blocks.remove(b_j)
                    else:
                        current_blocks.remove(b_i)
                        current_blocks.remove(b_j)
                        current_blocks.add(c_block)
                        
                    improved = True
                    found_merge = True
                    break
                    
            if found_merge:
                break
                
    # Convert to list of lists
    blocks = [list(all_k_blocks[i]) for i in current_blocks]
    
    return blocks
