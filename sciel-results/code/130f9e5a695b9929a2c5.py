import itertools
import numpy as np
import time
import random

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0
    
    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_tsubs = len(all_t_subsets)
    t_subset_to_index = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_blocks = len(all_k_subsets)
    
    # Map each block to the indices of t-subsets it covers
    block_to_tsubs = []
    for block in all_k_subsets:
        covered = []
        for tsub in itertools.combinations(block, t):
            idx = t_subset_to_index[tsub]
            covered.append(idx)
        block_to_tsubs.append(covered)
        
    # For faster lookup in SA, we might want to know which blocks cover which t-subsets
    # But for SA, we only need to update counts.
    
    # 2. Greedy Initialization
    uncovered = np.ones(num_tsubs, dtype=bool)
    gains = np.array([len(b) for b in block_to_tsubs], dtype=np.int32)
    
    selected_block_indices = set()
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Find max gain
        # Using argmax is fast, but ties are broken arbitrarily. 
        # We can add small random noise to gains to diversify
        noisy_gains = gains + np.random.rand(num_blocks) * 0.1
        b = np.argmax(noisy_gains)
        
        selected_block_indices.add(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                # Update gains for all blocks that cover this s_idx
                # This is the bottleneck of standard greedy.
                # For v=19, k=7, t=4, this is manageable.
                # We need tsub_to_blocks for this.
                
    # Wait, the standard greedy above is O(N_blocks * N_tsubs) per step if not careful.
    # Let's stick to the vectorized update or a simpler loop if N is small.
    # N_blocks = 50k, N_tsubs = 3876.
    # Steps ~ 150.
    # 150 * 50k * 35 (avg tsubs per block) is too slow for pure python loops.
    # We need the tsub_to_blocks mapping for fast gain updates.
    
    # Re-doing precomputation for Greedy
    tsub_to_blocks = [[] for _ in range(num_tsubs)]
    for b_idx, tsubs in enumerate(block_to_tsubs):
        for s_idx in tsubs:
            tsub_to_blocks[s_idx].append(b_idx)
            
    # Reset for Greedy
    uncovered = np.ones(num_tsubs, dtype=bool)
    gains = np.array([len(b) for b in block_to_tsubs], dtype=np.int32)
    selected_block_indices = set()
    
    while uncovered.any():
        if time.time() > deadline:
            break
        
        # Random tie-breaking
        max_gain = np.max(gains)
        candidates = np.where(gains == max_gain)[0]
        b = random.choice(candidates.tolist())
        
        selected_block_indices.add(b)
        
        for s_idx in block_to_tsubs[b]:
            if uncovered[s_idx]:
                uncovered[s_idx] = False
                for b2 in tsub_to_blocks[s_idx]:
                    gains[b2] -= 1

    if not uncovered.any():
        # 3. Simulated Annealing / Local Search
        # State: set of block indices
        # Cost: len(selected) + LAMBDA * num_uncovered
        
        LAMBDA = 100 # Large penalty for uncovered
        
        # Initialize cover counts
        cover_count = np.zeros(num_tsubs, dtype=np.int32)
        for b in selected_block_indices:
            for s in block_to_tsubs[b]:
                cover_count[s] += 1
                
        num_uncovered = np.sum(cover_count == 0)
        current_cost = len(selected_block_indices) + LAMBDA * num_uncovered
        
        best_selected = set(selected_block_indices)
        best_cost = current_cost
        best_uncovered = num_uncovered
        
        T = 50.0 # Initial temperature
        cooling_rate = 0.999
        min_T = 0.1
        
        # Precompute a list of all blocks for random selection
        all_block_list = list(range(num_blocks))
        
        while time.time() < deadline and T > min_T:
            # Generate a move
            move_type = random.random()
            
            if move_type < 0.3:
                # Remove a block
                if not selected_block_indices:
                    T *= cooling_rate
                    continue
                b_out = random.choice(list(selected_block_indices))
                
                # Calculate Delta
                delta_cost = 1 # +1 block removed
                delta_uncovered = 0
                for s in block_to_tsubs[b_out]:
                    if cover_count[s] == 1:
                        delta_uncovered += 1
                    cover_count[s] -= 1
                
                delta_cost += LAMBDA * delta_uncovered
                
                # Acceptance
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected_block_indices.remove(b_out)
                    current_cost += delta_cost
                    num_uncovered += delta_uncovered
                    
                    if num_uncovered == 0 and current_cost < best_cost:
                        best_selected = set(selected_block_indices)
                        best_cost = current_cost
                        best_uncovered = num_uncovered
                else:
                    # Revert
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] += 1
                        
            elif move_type < 0.7:
                # Add a block
                b_in = random.choice(all_block_list)
                if b_in in selected_block_indices:
                    T *= cooling_rate
                    continue
                    
                delta_cost = -1 # -1 block added
                delta_uncovered = 0
                for s in block_to_tsubs[b_in]:
                    if cover_count[s] == 0:
                        delta_uncovered -= 1
                    cover_count[s] += 1
                    
                delta_cost += LAMBDA * delta_uncovered
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected_block_indices.add(b_in)
                    current_cost += delta_cost
                    num_uncovered += delta_uncovered
                    
                    if num_uncovered == 0 and current_cost < best_cost:
                        best_selected = set(selected_block_indices)
                        best_cost = current_cost
                        best_uncovered = num_uncovered
                else:
                    # Revert
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] -= 1
                        
            else:
                # Swap: Remove one, Add one
                if not selected_block_indices:
                    T *= cooling_rate
                    continue
                b_out = random.choice(list(selected_block_indices))
                b_in = random.choice(all_block_list)
                if b_in in selected_block_indices:
                    T *= cooling_rate
                    continue
                    
                # Calculate Delta for b_out
                delta_cost_out = 1
                delta_unc_out = 0
                for s in block_to_tsubs[b_out]:
                    if cover_count[s] == 1:
                        delta_unc_out += 1
                    cover_count[s] -= 1
                delta_cost_out += LAMBDA * delta_unc_out
                
                # Calculate Delta for b_in (on the state AFTER b_out removed)
                delta_cost_in = -1
                delta_unc_in = 0
                for s in block_to_tsubs[b_in]:
                    if cover_count[s] == 0:
                        delta_unc_in -= 1
                    cover_count[s] += 1
                delta_cost_in += LAMBDA * delta_unc_in
                
                delta_cost = delta_cost_out + delta_cost_in
                delta_uncovered = delta_unc_out + delta_unc_in
                
                if delta_cost < 0 or random.random() < np.exp(-delta_cost / T):
                    selected_block_indices.remove(b_out)
                    selected_block_indices.add(b_in)
                    current_cost += delta_cost
                    num_uncovered += delta_uncovered
                    
                    if num_uncovered == 0 and current_cost < best_cost:
                        best_selected = set(selected_block_indices)
                        best_cost = current_cost
                        best_uncovered = num_uncovered
                else:
                    # Revert
                    for s in block_to_tsubs[b_in]:
                        cover_count[s] -= 1
                    for s in block_to_tsubs[b_out]:
                        cover_count[s] += 1
                        
            T *= cooling_rate
            
        # Return the best valid cover found
        if best_uncovered == 0:
            result = [list(all_k_subsets[i]) for i in best_selected]
            return result
        else:
            # Fallback to the initial greedy cover if SA failed to find valid
            # But SA usually finds valid ones quickly.
            # If we are here, we might have lost the best valid cover?
            # No, best_selected is only updated when valid.
            # If best_uncovered != 0, it means we never found a valid cover in SA?
            # But we started with a valid one. So best_uncovered should be 0 initially.
            # Unless the initial greedy failed?
            if not uncovered.any():
                 return [list(all_k_subsets[i]) for i in selected_block_indices]
            return []
    else:
        return []
