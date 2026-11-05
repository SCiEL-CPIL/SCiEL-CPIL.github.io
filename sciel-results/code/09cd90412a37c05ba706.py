import itertools
import numpy as np
from scipy.sparse import csr_matrix
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    n_t = len(t_subsets)
    subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Generate all k-blocks
    k_blocks = list(itertools.combinations(range(v), k))
    n_k = len(k_blocks)
    
    # 3. Precompute block_to_subsets and subset_to_blocks
    block_to_subsets = [[] for _ in range(n_k)]
    subset_to_blocks = [[] for _ in range(n_t)]
    
    for b_idx, block in enumerate(k_blocks):
        for t_sub in itertools.combinations(block, t):
            s_idx = subset_to_idx[t_sub]
            block_to_subsets[b_idx].append(s_idx)
            subset_to_blocks[s_idx].append(b_idx)
    
    # Convert to numpy arrays for fast operations
    block_subset_arrays = [np.array(subs, dtype=np.int32) for subs in block_to_subsets]
    subset_block_arrays = [np.array(blocks, dtype=np.int32) for blocks in subset_to_blocks]
    
    # 4. Build sparse coverage matrix for initial gain computation
    rows = []
    cols = []
    for b_idx in range(n_k):
        for s_idx in block_to_subsets[b_idx]:
            rows.append(b_idx)
            cols.append(s_idx)
    
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(n_k, n_t))
    
    def greedy_construct(seed):
        rng = np.random.default_rng(seed)
        uncovered = np.ones(n_t, dtype=bool)
        # Initial gains: count of subsets each block covers
        gains = np.array([len(subs) for subs in block_to_subsets], dtype=np.int32)
        
        selected_blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline - 2.0:
                break
                
            # Add noise for diversification
            noise = rng.uniform(0, 0.3, size=n_k)
            scores = gains.astype(np.float64) + noise
            best_idx = int(np.argmax(scores))
            
            if gains[best_idx] == 0:
                break
                
            selected_blocks.append(best_idx)
            
            # Update uncovered and gains using precomputed arrays
            b_subs = block_subset_arrays[best_idx]
            newly_covered = b_subs[uncovered[b_subs]]
            
            if len(newly_covered) == 0:
                continue
                
            uncovered[newly_covered] = False
            # Vectorized gain update
            for s_idx in newly_covered:
                blocks_covering = subset_block_arrays[s_idx]
                gains[blocks_covering] -= 1
        
        return [k_blocks[i] for i in selected_blocks]
    
    def prune_covering(blocks):
        if not blocks:
            return blocks
            
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for b in blocks:
            b_idx = k_blocks.index(b) if b in k_blocks else None
            if b_idx is None:
                # Find block index by value
                for i, kb in enumerate(k_blocks):
                    if kb == b:
                        b_idx = i
                        break
            if b_idx is not None:
                for s_idx in block_to_subsets[b_idx]:
                    cover_count[s_idx] += 1
        
        pruned_blocks = []
        for b in blocks:
            b_idx = k_blocks.index(b) if b in k_blocks else None
            if b_idx is None:
                for i, kb in enumerate(k_blocks):
                    if kb == b:
                        b_idx = i
                        break
            if b_idx is None:
                continue
            is_redundant = True
            for s_idx in block_to_subsets[b_idx]:
                if cover_count[s_idx] == 1:
                    is_redundant = False
                    break
            if not is_redundant:
                pruned_blocks.append(b)
            else:
                for s_idx in block_to_subsets[b_idx]:
                    cover_count[s_idx] -= 1
                    
        # Repeat a few times
        for _ in range(3):
            new_pruned = []
            for b in pruned_blocks:
                b_idx = k_blocks.index(b) if b in k_blocks else None
                if b_idx is None:
                    for i, kb in enumerate(k_blocks):
                        if kb == b:
                            b_idx = i
                            break
                if b_idx is None:
                    continue
                is_redundant = True
                for s_idx in block_to_subsets[b_idx]:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if not is_redundant:
                    new_pruned.append(b)
                else:
                    for s_idx in block_to_subsets[b_idx]:
                        cover_count[s_idx] -= 1
            pruned_blocks = new_pruned
            
        return pruned_blocks
    
    def local_search_with_swaps(blocks, time_limit, seed):
        """Local search with deletion and swap operations."""
        rng = random.Random(seed)
        current_blocks = list(blocks)
        
        if not current_blocks:
            return current_blocks
            
        # Build coverage counts
        cover_count = np.zeros(n_t, dtype=np.int32)
        block_indices = []
        for b in current_blocks:
            b_idx = None
            for i, kb in enumerate(k_blocks):
                if kb == b:
                    b_idx = i
                    break
            if b_idx is not None:
                block_indices.append(b_idx)
                for s_idx in block_to_subsets[b_idx]:
                    cover_count[s_idx] += 1
        
        current_blocks_set = set(block_indices)
        improved = True
        start_time_ls = time.time()
        
        while improved and (time.time() - start_time_ls) < time_limit:
            improved = False
            
            # Deletion pass
            indices = list(range(len(block_indices)))
            rng.shuffle(indices)
            
            for i in indices:
                if i >= len(block_indices):
                    continue
                if (time.time() - start_time_ls) > time_limit:
                    break
                b_idx = block_indices[i]
                is_redundant = True
                b_subs = block_subset_arrays[b_idx]
                for s_idx in b_subs:
                    if cover_count[s_idx] == 1:
                        is_redundant = False
                        break
                if is_redundant:
                    for s_idx in b_subs:
                        cover_count[s_idx] -= 1
                    current_blocks_set.discard(b_idx)
                    block_indices.pop(i)
                    improved = True
                    break  # Restart scan after removal
            
            # Swap pass
            if (time.time() - start_time_ls) > time_limit:
                break
                
            indices = list(range(len(block_indices)))
            rng.shuffle(indices)
            
            swaps_done = 0
            max_swaps = 30
            
            for i in indices:
                if (time.time() - start_time_ls) > time_limit:
                    break
                if swaps_done >= max_swaps:
                    break
                if i >= len(block_indices):
                    continue
                    
                old_block = block_indices[i]
                old_subs = block_subset_arrays[old_block]
                
                # Find unique subsets covered by old_block
                unique_subs = []
                for s_idx in old_subs:
                    if cover_count[s_idx] == 1:
                        unique_subs.append(s_idx)
                
                if not unique_subs:
                    # Block is not critical - try swapping with a random block
                    new_block = rng.randint(0, n_k - 1)
                    if new_block == old_block:
                        continue
                    if new_block in current_blocks_set:
                        continue
                        
                    # Perform swap
                    for s_idx in old_subs:
                        cover_count[s_idx] -= 1
                    new_subs = block_subset_arrays[new_block]
                    for s_idx in new_subs:
                        cover_count[s_idx] += 1
                    current_blocks_set.discard(old_block)
                    current_blocks_set.add(new_block)
                    block_indices[i] = new_block
                    swaps_done += 1
                    improved = True
                else:
                    # Block is critical - find candidates that cover all unique_subs
                    candidates = set(subset_block_arrays[unique_subs[0]])
                    for s_idx in unique_subs[1:]:
                        candidates &= set(subset_block_arrays[s_idx])
                    candidates.discard(old_block)
                    candidates -= current_blocks_set
                    
                    if candidates:
                        new_block = list(candidates)[rng.randint(0, len(candidates) - 1)]
                        # Perform swap
                        for s_idx in old_subs:
                            cover_count[s_idx] -= 1
                        new_subs = block_subset_arrays[new_block]
                        for s_idx in new_subs:
                            cover_count[s_idx] += 1
                        current_blocks_set.discard(old_block)
                        current_blocks_set.add(new_block)
                        block_indices[i] = new_block
                        swaps_done += 1
                        improved = True
        
        return [k_blocks[i] for i in block_indices]
    
    best_covering = None
    best_size = float('inf')
    
    # Run multiple greedy restarts
    n_restarts = 15
    
    for i in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
            
        covering = greedy_construct(i)
        covering = prune_covering(covering)
        
        # Local search with swaps
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 2.0:
            covering = local_search_with_swaps(covering, min(remaining_time, 10.0), seed=i)
            covering = prune_covering(covering)
        
        if len(covering) < best_size:
            best_size = len(covering)
            best_covering = covering
    
    if best_covering is None:
        best_covering = greedy_construct(0)
        best_covering = prune_covering(best_covering)
        
    return best_covering
