import itertools
import time
import numpy as np
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 112.0  # Safety margin for return
    
    # Precompute all t-subsets
    all_subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all k-blocks
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # For each block, store the list of subset indices it covers
    block_to_subsets = [[] for _ in range(n_blocks)]
    # For each subset, store which blocks cover it
    subset_to_blocks = [[] for _ in range(n_subsets)]
    
    for b_idx, block in enumerate(all_blocks):
        for subset in itertools.combinations(block, t):
            u_idx = subset_to_idx[subset]
            block_to_subsets[b_idx].append(u_idx)
            subset_to_blocks[u_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    block_subsets_arr = [np.array(bs, dtype=np.int32) for bs in block_to_subsets]
    subset_blocks_arr = [np.array(sb, dtype=np.int32) for sb in subset_to_blocks]
    
    # Frequency of each subset across all blocks (for weighted greedy)
    freq = np.zeros(n_subsets, dtype=np.int32)
    for u in range(n_subsets):
        freq[u] = len(subset_to_blocks[u])
    inv_freq = 1.0 / (freq + 1e-9)
    
    best_blocks = None
    best_len = float('inf')
    
    # Multiple strategies with different random seeds
    seed = 0
    while time.time() < deadline - 15:  # Leave time for local search
        strategy = seed % 4
        
        if strategy == 0:
            blocks = _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, 
                                   n_subsets, deadline, mode='max_gain')
        elif strategy == 1:
            blocks = _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, 
                                   n_subsets, deadline, mode='weighted', inv_freq=inv_freq)
        elif strategy == 2:
            blocks = _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, 
                                   n_subsets, deadline, mode='ratio', inv_freq=inv_freq)
        else:
            blocks = _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, 
                                   n_subsets, deadline, mode='randomized', 
                                   inv_freq=inv_freq, seed=seed)
        
        if blocks is not None:
            # Remove redundant blocks
            blocks = _remove_redundant(blocks, all_blocks, block_to_subsets, subset_to_blocks, n_subsets)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks
            
            # Early stop if we hit a very good number
            if best_len <= 76:
                break
        
        seed += 1
    
    # Local search on the best solution found
    if best_blocks is not None and time.time() < deadline - 5:
        best_blocks = _local_search(best_blocks, all_blocks, block_to_subsets, 
                                     subset_to_blocks, n_subsets, deadline)
    
    if best_blocks is None:
        # Fallback: deterministic greedy
        best_blocks = _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, 
                                     n_subsets, deadline, mode='max_gain')
        if best_blocks is None:
            best_blocks = []
    
    return best_blocks

def _greedy_cover(all_blocks, block_to_subsets, subset_to_blocks, n_subsets, deadline, 
                  mode='max_gain', inv_freq=None, seed=0):
    """
    Greedy set cover with different selection modes.
    mode: 'max_gain' (max new coverage), 'weighted' (max weighted new coverage),
          'ratio' (max gain per point), 'randomized' (random from top candidates)
    """
    uncovered = np.ones(n_subsets, dtype=bool)
    selected_indices = []
    used_blocks = set()
    
    # Initialize gains: for each block, number of uncovered subsets it covers
    gains = np.zeros(len(all_blocks), dtype=np.int32)
    for u in range(n_subsets):
        for b in subset_to_blocks[u]:
            gains[b] += 1
    
    while uncovered.any():
        if time.time() > deadline:
            return None
        
        if mode == 'max_gain':
            # Find block with max gain among unused blocks
            max_gain = -1
            best_idx = -1
            for b in range(len(all_blocks)):
                if b not in used_blocks and gains[b] > max_gain:
                    max_gain = gains[b]
                    best_idx = b
            
            if max_gain <= 0:
                break
                
            selected_indices.append(best_idx)
            used_blocks.add(best_idx)
            
            # Update uncovered and gains
            for u in block_to_subsets[best_idx]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1
                        
        elif mode == 'weighted':
            # Weighted gain = sum of inv_freq for uncovered subsets in block
            max_wgain = -1.0
            best_idx = -1
            for b in range(len(all_blocks)):
                if b not in used_blocks:
                    wgain = 0.0
                    for u in block_to_subsets[b]:
                        if uncovered[u]:
                            wgain += inv_freq[u]
                    if wgain > max_wgain:
                        max_wgain = wgain
                        best_idx = b
            
            if max_wgain <= 0:
                break
                
            selected_indices.append(best_idx)
            used_blocks.add(best_idx)
            
            # Update uncovered and gains
            for u in block_to_subsets[best_idx]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1
                        
        elif mode == 'ratio':
            # Ratio = gain / number of points in block (penalizes large blocks)
            # All blocks have same size k, so this is equivalent to max_gain
            # But we can use a modified ratio: gain / (k * overlap_factor)
            max_ratio = -1.0
            best_idx = -1
            for b in range(len(all_blocks)):
                if b not in used_blocks and gains[b] > 0:
                    # Calculate overlap with already selected blocks
                    overlap = 0
                    for u in block_to_subsets[b]:
                        if not uncovered[u]:
                            overlap += 1
                    # Ratio: new coverage / total coverage potential
                    total_cov = len(block_to_subsets[b])
                    if total_cov > 0:
                        ratio = gains[b] / (total_cov + overlap * 0.5)
                    else:
                        ratio = 0
                    if ratio > max_ratio:
                        max_ratio = ratio
                        best_idx = b
            
            if max_ratio <= 0:
                break
                
            selected_indices.append(best_idx)
            used_blocks.add(best_idx)
            
            # Update uncovered and gains
            for u in block_to_subsets[best_idx]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1
                        
        elif mode == 'randomized':
            # Calculate gains, then pick randomly from top candidates
            max_gain = 0
            for b in range(len(all_blocks)):
                if b not in used_blocks and gains[b] > max_gain:
                    max_gain = gains[b]
            
            if max_gain <= 0:
                break
            
            # Get indices of blocks with max gain
            candidates = [b for b in range(len(all_blocks)) 
                         if b not in used_blocks and gains[b] == max_gain]
            
            # If too many candidates, sample a few
            rng = random.Random(seed + len(selected_indices))
            if len(candidates) > 10:
                candidates = rng.sample(candidates, 10)
            
            best_idx = rng.choice(candidates)
            selected_indices.append(best_idx)
            used_blocks.add(best_idx)
            
            # Update uncovered and gains
            for u in block_to_subsets[best_idx]:
                if uncovered[u]:
                    uncovered[u] = False
                    for b in subset_to_blocks[u]:
                        gains[b] -= 1
        else:
            raise ValueError("Unknown mode")
        
        # Early exit if all covered
        if not uncovered.any():
            break
    
    if uncovered.any():
        # Incomplete covering
        return None
    
    return [list(all_blocks[i]) for i in selected_indices]

def _remove_redundant(blocks, all_blocks, block_to_subsets, subset_to_blocks, n_subsets):
    """
    Remove blocks that do not cover any unique t-subset.
    """
    if not blocks:
        return blocks
    
    # Map from block tuple to its index in all_blocks
    block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks]
    
    # Count how many selected blocks cover each subset
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        for u in block_to_subsets[b_idx]:
            cover_count[u] += 1
    
    # Try to remove blocks that don't reduce coverage below 1
    improved = True
    while improved:
        improved = False
        for i in range(len(selected_block_indices) - 1, -1, -1):
            b_idx = selected_block_indices[i]
            # Check if this block is the only one covering any subset
            redundant = True
            for u in block_to_subsets[b_idx]:
                if cover_count[u] == 1:
                    redundant = False
                    break
            if redundant:
                # Remove this block
                selected_block_indices.pop(i)
                # Update cover_count
                for u in block_to_subsets[b_idx]:
                    cover_count[u] -= 1
                improved = True
                break  # Restart the loop since indices changed
    
    return [list(all_blocks[b_idx]) for b_idx in selected_block_indices]

def _local_search(blocks, all_blocks, block_to_subsets, subset_to_blocks, n_subsets, deadline):
    """
    Local search: try to swap blocks to reduce total count.
    """
    if not blocks:
        return blocks
    
    # Rebuild structures for the current solution
    block_list_to_idx = {tuple(b): i for i, b in enumerate(all_blocks)}
    selected_block_indices = [block_list_to_idx[tuple(b)] for b in blocks]
    
    # Count how many selected blocks cover each subset
    cover_count = np.zeros(n_subsets, dtype=np.int32)
    for b_idx in selected_block_indices:
        for u in block_to_subsets[b_idx]:
            cover_count[u] += 1
    
    # Try to find a block that can be removed and its unique subsets covered by adding one block
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        # Try each selected block
        for i in range(len(selected_block_indices)):
            if time.time() > deadline - 3:
                break
            b_idx = selected_block_indices[i]
            
            # Find subsets that would become uncovered if we remove this block
            unique_subsets = []
            for u in block_to_subsets[b_idx]:
                if cover_count[u] == 1:
                    unique_subsets.append(u)
            
            if not unique_subsets:
                # This block is redundant, remove it
                selected_block_indices.pop(i)
                for u in block_to_subsets[b_idx]:
                    cover_count[u] -= 1
                improved = True
                break
            
            # Try to find a single block that covers all unique_subsets
            # and is not already in the solution
            best_candidate = -1
            best_gain = 0
            for u in unique_subsets:
                for b in subset_to_blocks[u]:
                    if b not in selected_block_indices:
                        # Check if this block covers all unique_subsets
                        covers_all = True
                        for u2 in unique_subsets:
                            if u2 not in block_to_subsets[b]:
                                covers_all = False
                                break
                        if covers_all:
                            # Count how many other subsets this block covers
                            gain = 0
                            for u2 in block_to_subsets[b]:
                                if cover_count[u2] == 0:
                                    gain += 1
                            if gain > best_gain:
                                best_gain = gain
                                best_candidate = b
            
            if best_candidate >= 0:
                # Remove the old block and add the new one
                selected_block_indices.pop(i)
                for u in block_to_subsets[b_idx]:
                    cover_count[u] -= 1
                selected_block_indices.append(best_candidate)
                for u in block_to_subsets[best_candidate]:
                    cover_count[u] += 1
                improved = True
                break
    
    return [list(all_blocks[b_idx]) for b_idx in selected_block_indices]
