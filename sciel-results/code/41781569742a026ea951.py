import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0

    # 1. Precomputation
    # Map t-subsets to indices 0..N_t-1
    all_t_subsets = list(itertools.combinations(range(v), t))
    N_t = len(all_t_subsets)
    t_subset_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
    
    # Generate all k-subsets
    all_k_subsets = list(itertools.combinations(range(v), k))
    N_k = len(all_k_subsets)
    
    # Precompute mask and t-indices for each k-block
    block_masks = [0] * N_k
    block_t_indices = [[] for _ in range(N_k)]
    
    for i, block in enumerate(all_k_subsets):
        mask = 0
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            mask |= (1 << idx)
            indices.append(idx)
        block_masks[i] = mask
        block_t_indices[i] = indices

    initial_uncovered_mask = (1 << N_t) - 1
    best_blocks = None
    best_count = float('inf')

    # 2. Main Loop
    while time.time() < deadline:
        uncovered_mask = initial_uncovered_mask
        selected_blocks = []
        cover_counts = [0] * N_t
        zero_count = N_t # Number of t-subsets with count 0

        # Greedy Phase
        while uncovered_mask:
            max_gain = 0
            best_indices = []
            
            # Calculate gains
            masks = block_masks
            for i in range(N_k):
                gain = (uncovered_mask & masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_indices = [i]
                elif gain == max_gain:
                    best_indices.append(i)
            
            if not best_indices:
                break # Should not happen if covering exists
                
            # Randomize among top candidates
            if len(best_indices) > 50:
                chosen_idx = random.choice(best_indices)
            else:
                chosen_idx = random.choice(best_indices)
                
            selected_blocks.append(chosen_idx)
            uncovered_mask &= ~block_masks[chosen_idx]
            
            # Update counts and zero_count
            for idx in block_t_indices[chosen_idx]:
                old_count = cover_counts[idx]
                new_count = old_count + 1
                cover_counts[idx] = new_count
                if old_count == 0 and new_count == 1:
                    zero_count -= 1

        # Local Search: Redundancy Removal
        # Try to remove blocks that are no longer needed
        improved = True
        while improved:
            improved = False
            # Iterate in reverse to handle index shifts easily if we pop
            for i in range(len(selected_blocks) - 1, -1, -1):
                block_idx = selected_blocks[i]
                
                # Tentatively remove
                for t_idx in block_t_indices[block_idx]:
                    old_count = cover_counts[t_idx]
                    new_count = old_count - 1
                    cover_counts[t_idx] = new_count
                    if old_count == 1 and new_count == 0:
                        zero_count += 1
                    
                # Check if any t-subset is now uncovered (count 0)
                if zero_count == 0:
                    # Safe to remove
                    selected_blocks.pop(i)
                    improved = True
                    break # Restart loop as indices changed
                else:
                    # Restore
                    for t_idx in block_t_indices[block_idx]:
                        old_count = cover_counts[t_idx]
                        new_count = old_count + 1
                        cover_counts[t_idx] = new_count
                        if old_count == 0 and new_count == 1:
                            zero_count -= 1

        if len(selected_blocks) < best_count:
            best_count = len(selected_blocks)
            best_blocks = [list(all_k_subsets[i]) for i in selected_blocks]

    if best_blocks is None:
        # Fallback: Deterministic Greedy
        uncovered_mask = initial_uncovered_mask
        selected_blocks = []
        while uncovered_mask:
            max_gain = 0
            best_idx = 0
            for i in range(N_k):
                gain = (uncovered_mask & block_masks[i]).bit_count()
                if gain > max_gain:
                    max_gain = gain
                    best_idx = i
            selected_blocks.append(best_idx)
            uncovered_mask &= ~block_masks[best_idx]
        best_blocks = [list(all_k_subsets[i]) for i in selected_blocks]

    return best_blocks
