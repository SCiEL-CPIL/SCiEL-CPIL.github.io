import itertools
import time
import random
import numpy as np

# Precomputation for v=19, k=7, t=4
v, k, t = 19, 7, 4
subsets_4 = list(itertools.combinations(range(v), t))
subset_to_idx = {s: i for i, s in enumerate(subsets_4)}
N_subsets = len(subsets_4)

blocks_7 = list(itertools.combinations(range(v), k))
block_masks = []
block_subsets = []
for b in blocks_7:
    mask = 0
    idxs = []
    for s in itertools.combinations(b, t):
        idx = subset_to_idx[s]
        mask |= (1 << idx)
        idxs.append(idx)
    block_masks.append(mask)
    block_subsets.append(idxs)

N_blocks = len(blocks_7)

# Convert to numpy for faster weighted sum if needed
block_subsets_np = [np.array(idxs, dtype=np.int32) for idxs in block_subsets]

def fast_greedy(deadline):
    """
    Standard greedy using bitsets. Very fast.
    """
    full_mask = (1 << N_subsets) - 1
    uncovered = full_mask
    used_blocks = []
    remaining = N_subsets
    
    while remaining > 0 and time.time() < deadline:
        best_gain = 0
        best_idx = -1
        
        # Scan all blocks
        for i in range(N_blocks):
            # If we have used this block, skip? 
            # In set cover, we generally don't reuse blocks if distinctness is required, 
            # but the problem says "blocks must be distinct". 
            # So we should avoid picking the same block twice.
            # However, if we pick a block, it covers some subsets. 
            # We just need to track used blocks.
            if i in used_blocks:
                continue
            
            common = uncovered & block_masks[i]
            gain = common.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_idx = i
        
        if best_idx == -1 or best_gain == 0:
            break
        
        used_blocks.append(best_idx)
        uncovered &= ~block_masks[best_idx]
        remaining -= best_gain
        
    return used_blocks, uncovered

def local_search(used_blocks, deadline):
    """
    Improve the cover by removing redundancies and swapping blocks.
    """
    if not used_blocks:
        return used_blocks
    
    # Build coverage counts
    cover_counts = [0] * N_subsets
    for idx in used_blocks:
        for s_idx in block_subsets[idx]:
            cover_counts[s_idx] += 1
            
    used_set = set(used_blocks)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Redundancy Removal
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for idx in used_blocks:
                is_redundant = True
                for s_idx in block_subsets[idx]:
                    if cover_counts[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in block_subsets[idx]:
                        cover_counts[s_idx] -= 1
                    used_set.remove(idx)
                    changed = True
                    improved = True
                else:
                    new_blocks.append(idx)
            used_blocks = new_blocks
            
        if not used_blocks:
            break

        # 2. 1-for-1 Swaps
        # Identify blocks with unique subsets
        unique_subsets_map = {}
        for idx in used_blocks:
            u_list = [s_idx for s_idx in block_subsets[idx] if cover_counts[s_idx] == 1]
            if u_list:
                unique_subsets_map[idx] = u_list

        found_swap = False
        # Shuffle to randomize order
        block_order = list(used_blocks)
        random.shuffle(block_order)
        
        for i_idx in block_order:
            if i_idx not in unique_subsets_map:
                continue
            u_list = unique_subsets_map[i_idx]
            u_set = set(u_list)
            
            # Try to find a block j not in used_blocks that covers u_set
            for j_idx in range(N_blocks):
                if j_idx in used_set:
                    continue
                
                # Check if j covers all unique subsets of i
                covers = True
                for s_idx in u_list:
                    if s_idx not in block_subsets[j_idx]:
                        covers = False
                        break
                
                if covers:
                    # Perform swap
                    used_blocks.remove(i_idx)
                    used_set.remove(i_idx)
                    for s_idx in block_subsets[i_idx]:
                        cover_counts[s_idx] -= 1
                    
                    used_blocks.append(j_idx)
                    used_set.add(j_idx)
                    for s_idx in block_subsets[j_idx]:
                        cover_counts[s_idx] += 1
                    
                    found_swap = True
                    improved = True
                    break
            
            if found_swap:
                break # Restart loop after a swap
        
        # 3. 2-for-1 Swaps (Replace two blocks with one)
        # This is more expensive but can significantly reduce size
        if time.time() < deadline:
            # Get current unique subsets for each block
            # For a pair (A, B), we want to find C such that C covers (Unique(A) U Unique(B))
            # And C is not in used_blocks.
            
            # To make this efficient, we only consider pairs where A and B have small unique sets
            # Or we just iterate over a random sample of pairs.
            
            # Let's try a simpler version:
            # For each block i, find its unique subsets U_i.
            # If U_i is small, try to merge it with another block j.
            
            # Actually, a simpler heuristic:
            # Try to replace any two blocks {i, j} with a single block k 
            # if k covers all subsets that are uniquely covered by i or j.
            
            # This is O(N^3) which is too slow.
            # We will skip 2-for-1 for now to keep it fast, or implement a limited version.
            
            # Limited 2-for-1:
            # Pick a random block i.
            # Find its unique subsets U_i.
            # If len(U_i) > 0:
            #   Pick a random block j.
            #   Find U_j.
            #   Target = U_i U U_j.
            #   Find k not in used that covers Target.
            
            if len(used_blocks) > 1:
                i_idx = random.choice(used_blocks)
                u_i = [s_idx for s_idx in block_subsets[i_idx] if cover_counts[s_idx] == 1]
                
                if u_i:
                    # Find a partner j
                    j_idx = random.choice([x for x in used_blocks if x != i_idx])
                    u_j = [s_idx for s_idx in block_subsets[j_idx] if cover_counts[s_idx] == 1]
                    
                    target_set = set(u_i + u_j)
                    
                    # Find k that covers target_set
                    for k_idx in range(N_blocks):
                        if k_idx in used_set:
                            continue
                        
                        covers = True
                        for s_idx in target_set:
                            if s_idx not in block_subsets[k_idx]:
                                covers = False
                                break
                        
                        if covers:
                            # Perform 2-for-1 swap
                            used_blocks.remove(i_idx)
                            used_blocks.remove(j_idx)
                            used_set.remove(i_idx)
                            used_set.remove(j_idx)
                            
                            for s_idx in block_subsets[i_idx]:
                                cover_counts[s_idx] -= 1
                            for s_idx in block_subsets[j_idx]:
                                cover_counts[s_idx] -= 1
                                
                            used_blocks.append(k_idx)
                            used_set.add(k_idx)
                            for s_idx in block_subsets[k_idx]:
                                cover_counts[s_idx] += 1
                            
                            improved = True
                            break

    return used_blocks

def solve():
    deadline = time.time() + 118
    best_blocks = None
    best_len = float('inf')
    
    # Initial weights for Lagrangian (if we use it)
    w = np.ones(N_subsets, dtype=np.float64)
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        remaining_time = deadline - time.time()
        
        # 1. Generate initial solution using fast greedy
        used_blocks, uncovered = fast_greedy(deadline - 1)
        
        if len(used_blocks) >= best_len:
            continue
            
        # 2. Local Search
        used_blocks = local_search(used_blocks, deadline - 5)
        
        if len(used_blocks) < best_len:
            best_len = len(used_blocks)
            best_blocks = [blocks_7[i] for i in used_blocks]
            if best_len <= 131:
                break
        
        # 3. Update weights based on uncovered subsets (if any)
        # If the cover is invalid (uncovered != 0), we need to fix it.
        # But fast_greedy should produce a valid cover if it finishes.
        # If it breaks early due to time, it might be invalid.
        # We should ensure validity.
        
        # Check validity
        full_mask = (1 << N_subsets) - 1
        current_mask = 0
        for idx in used_blocks:
            current_mask |= block_masks[idx]
        
        if current_mask != full_mask:
            # Invalid cover. Try to repair by adding blocks for uncovered subsets
            # This is a fallback. Ideally, greedy should finish.
            # If we are close to deadline, we might not have time.
            # For now, if invalid, we discard this candidate.
            continue
            
        # Update weights to prioritize scarce subsets in the next iteration
        # Count how many times each subset is covered
        cover_counts = [0] * N_subsets
        for idx in used_blocks:
            for s_idx in block_subsets[idx]:
                cover_counts[s_idx] += 1
                
        # Increase weight for subsets that are barely covered (count == 1)
        # This encourages future greedy picks to cover them more robustly
        for s_idx in range(N_subsets):
            if cover_counts[s_idx] == 1:
                w[s_idx] += 0.1
            elif cover_counts[s_idx] > 1:
                w[s_idx] = max(0.1, w[s_idx] * 0.9)
                
        # Normalize
        w_sum = np.sum(w)
        if w_sum > 0:
            w = w * (N_subsets / w_sum)
            
        # Occasionally reset weights to escape local minima
        if iteration % 5 == 0:
            w = np.ones(N_subsets, dtype=np.float64)
            w += np.random.uniform(-0.1, 0.1, N_subsets)
            w = np.maximum(0.1, w)
            w = w / np.sum(w) * N_subsets

    return best_blocks

def construct(v, k, t):
    if v == 19 and k == 7 and t == 4:
        return solve()
    else:
        # Fallback to simple greedy for other parameters
        uncovered = set(itertools.combinations(range(v), t))
        blocks = []
        while uncovered:
            best_block = None
            max_covered = 0
            for b in itertools.combinations(range(v), k):
                covered = 0
                for s in itertools.combinations(b, t):
                    if s in uncovered:
                        covered += 1
                if covered > max_covered:
                    max_covered = covered
                    best_block = b
            if not best_block:
                break
            blocks.append(list(best_block))
            for s in itertools.combinations(best_block, t):
                uncovered.discard(s)
        return blocks
