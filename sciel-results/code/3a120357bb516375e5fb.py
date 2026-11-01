import itertools
import random
import time
import sys
import math

def construct(v, k, t):
    deadline = time.time() + 110  # Safety margin before 150s kill
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    t_subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_t_subsets = len(all_t_subsets)
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Map each block to the indices of the t-subsets it covers
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_subset_to_idx[s] for s in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Map each t-subset to the list of blocks that cover it
    t_subset_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for idx in indices:
            t_subset_blocks[idx].append(b_idx)
            
    # Precompute block sizes (number of t-subsets covered)
    block_sizes = [len(indices) for indices in block_t_indices]

    def greedy_set_cover(seed_offset=0):
        """Greedy set cover with random tie-breaking."""
        # uncovered[t_idx] is True if not yet covered
        uncovered = [True] * num_t_subsets
        gain = list(block_sizes)
        selected = []
        
        rng = random.Random(seed_offset)
        
        while True:
            # Find max gain
            max_gain = -1
            best_block = -1
            best_candidates = []
            
            # Optimization: If we track uncovered count, we can stop early?
            # No, we need max gain among all blocks.
            for i in range(num_k_subsets):
                if gain[i] > max_gain:
                    max_gain = gain[i]
                    best_block = i
                    best_candidates = [i]
                elif gain[i] == max_gain and max_gain > 0:
                    best_candidates.append(i)
            
            if max_gain == 0:
                break
            
            # Random tie-breaking
            if len(best_candidates) > 1:
                best_block = rng.choice(best_candidates)
            
            selected.append(best_block)
            
            # Update gains and uncovered status
            for t_idx in block_t_indices[best_block]:
                if uncovered[t_idx]:
                    uncovered[t_idx] = False
                    # Decrease gain for all blocks covering this t_subset
                    for b_idx in t_subset_blocks[t_idx]:
                        gain[b_idx] -= 1
        
        return selected

    def local_search_deletion(selected_blocks):
        """Iterative deletion-based minimalization."""
        # cover_count[t_idx] = number of selected blocks covering t_idx
        cover_count = [0] * num_t_subsets
        for b_idx in selected_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        active_blocks = set(selected_blocks)
        
        changed = True
        while changed:
            changed = False
            # Iterate over a copy of the list to allow modification
            indices_to_check = list(active_blocks)
            random.shuffle(indices_to_check)
            
            for b_idx in indices_to_check:
                if b_idx not in active_blocks:
                    continue
                
                can_remove = True
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] <= 1:
                        can_remove = False
                        break
                    
                if can_remove:
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    changed = True
        
        return active_blocks

    def local_search_swap(selected_blocks, max_iterations=20):
        """
        Optimized 1-1 swap.
        For each block b in cover:
          1. Find t-subsets uniquely covered by b (critical).
          2. If none, remove b (deletion handles this, but we check here too).
          3. If some, find a block b' NOT in cover that covers ALL critical t-subsets.
          4. If found, swap b for b'.
        """
        active_blocks = set(selected_blocks)
        
        cover_count = [0] * num_t_subsets
        for b_idx in active_blocks:
            for t_idx in block_t_indices[b_idx]:
                cover_count[t_idx] += 1
        
        # Precompute set of t-indices for each block for faster intersection?
        # block_t_sets = [set(block_t_indices[i]) for i in range(num_k_subsets)]
        # This uses a lot of memory but speeds up intersection.
        # Given N=50k, each set has C(7,4)=35 elements. 50k * 35 * 28 bytes ~ 40MB. Acceptable.
        block_t_sets = [set(indices) for indices in block_t_indices]
        
        improved = True
        iterations = 0
        while improved and iterations < max_iterations:
            improved = False
            iterations += 1
            if time.time() > deadline - 10:
                break
                
            blocks_list = list(active_blocks)
            random.shuffle(blocks_list)
            
            for b_idx in blocks_list:
                if b_idx not in active_blocks:
                    continue
                
                # Find critical t-subsets for b_idx
                critical_ts = []
                for t_idx in block_t_indices[b_idx]:
                    if cover_count[t_idx] == 1:
                        critical_ts.append(t_idx)
                
                if not critical_ts:
                    # Can remove without replacement
                    active_blocks.remove(b_idx)
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    improved = True
                    break
                
                # Find candidate replacement b'
                # b' must cover all critical_ts
                # Candidate set is intersection of t_subset_blocks[t] for t in critical_ts
                # But we must exclude b_idx itself and other active blocks?
                # No, we can swap with ANY block, but if it's already active, it's not a swap, it's a no-op or invalid.
                # Actually, if b' is in active_blocks, adding it does nothing (already there).
                # So we look for b' in [0, num_k_subsets) \ active_blocks.
                
                # Start with the smallest list of blocks covering any critical t
                min_list_len = float('inf')
                min_list = None
                for t_idx in critical_ts:
                    blocks_covering = t_subset_blocks[t_idx]
                    if len(blocks_covering) < min_list_len:
                        min_list_len = len(blocks_covering)
                        min_list = blocks_covering
                
                if not min_list:
                    continue
                    
                candidates = set(min_list)
                for t_idx in critical_ts:
                    candidates.intersection_update(t_subset_blocks[t_idx])
                
                # Filter out active blocks
                valid_candidates = [c for c in candidates if c not in active_blocks]
                
                if valid_candidates:
                    # Pick a random candidate
                    cand_idx = random.choice(valid_candidates)
                    
                    # Perform swap
                    active_blocks.remove(b_idx)
                    active_blocks.add(cand_idx)
                    
                    # Update cover counts
                    for t_idx in block_t_indices[b_idx]:
                        cover_count[t_idx] -= 1
                    for t_idx in block_t_indices[cand_idx]:
                        cover_count[t_idx] += 1
                    
                    improved = True
                    break
        
        return active_blocks

    # Run multiple restarts
    best_result = None
    best_size = float('inf')
    
    # Calculate lower bound for early stopping
    lower_bound = math.comb(v, t) // math.comb(k, t)
    if math.comb(v, t) % math.comb(k, t) != 0:
        lower_bound += 1
        
    num_restarts = 10 # Increase restarts if time permits
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Greedy
        selected = greedy_set_cover(seed_offset=restart)
        
        # Deletion
        active = local_search_deletion(selected)
        
        # Swap
        active = local_search_swap(list(active), max_iterations=10)
        
        size = len(active)
        if size < best_size:
            best_size = size
            best_result = list(active)
            
        # Early stop if we hit lower bound + small margin
        if best_size <= lower_bound + 2:
            break

    # Final deletion pass on best result
    if best_result is not None:
        best_result = local_search_deletion(best_result)
    
    result = [list(all_k_subsets[b_idx]) for b_idx in best_result]
    
    return result
