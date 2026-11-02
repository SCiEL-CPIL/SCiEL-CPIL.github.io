"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    deadline = time.time() + 110.0  # Safety margin
    
    # 1. Precompute all t-subsets and map them to indices
    t_sets = list(itertools.combinations(range(v), t))
    t_set_index = {ts: i for i, ts in enumerate(t_sets)}
    num_t_sets = len(t_sets)
    
    # 2. Precompute all k-blocks and the t-sets they cover
    all_blocks = list(itertools.combinations(range(v), k))
    block_cover_indices = []
    for block in all_blocks:
        indices = []
        for ts in itertools.combinations(block, t):
            indices.append(t_set_index[ts])
        block_cover_indices.append(indices)
    
    num_blocks = len(all_blocks)
    
    # Precompute reverse mapping: for each t-set, which blocks cover it
    t_set_to_blocks = [[] for _ in range(num_t_sets)]
    for b_idx in range(num_blocks):
        for ts_idx in block_cover_indices[b_idx]:
            t_set_to_blocks[ts_idx].append(b_idx)
    
    # Convert to numpy arrays for faster operations
    block_cover_indices_arr = [np.array(idx, dtype=np.int32) for idx in block_cover_indices]
    
    # 3. Get initial solution via greedy
    best_solution = _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices_arr, num_t_sets, deadline, rng_seed=42, randomize=False, t_set_to_blocks=t_set_to_blocks)
    if not best_solution:
        return []
    
    best_size = len(best_solution)
    
    # 4. Local search / WalkSAT-style improvement
    # Objective: minimize number of blocks
    # We try to remove blocks if possible (if their t-sets are covered by other selected blocks)
    # Or swap blocks to reduce count
    
    # Represent solution as set of block indices for fast lookup
    selected_block_indices = set()
    for i, block in enumerate(best_solution):
        # Find the index of this block in all_blocks
        # This is slow, so let's store indices directly
        pass
    
    # Better approach: store indices from the start
    # Let's modify _greedy_cover to return indices
    
    # Actually, let's just work with the list of blocks and use a set of frozensets for membership
    # But that's slow. Let's refactor to use indices.
    
    # Let's re-run greedy but track indices
    best_solution, best_indices = _greedy_cover_with_indices(v, k, t, t_set_index, all_blocks, block_cover_indices_arr, num_t_sets, deadline, rng_seed=42, randomize=False, t_set_to_blocks=t_set_to_blocks)
    
    if not best_solution:
        return []
    
    best_size = len(best_solution)
    
    # Local search loop
    improved = True
    while improved and time.time() < deadline - 5.0:
        improved = False
        # Try to remove each block
        # For each selected block, check if all its t-sets are covered by other selected blocks
        # If so, remove it.
        
        # To speed up, we can maintain a count of how many selected blocks cover each t-set
        cover_count = np.zeros(num_t_sets, dtype=np.int32)
        for b_idx in best_indices:
            for ts_idx in block_cover_indices_arr[b_idx]:
                cover_count[ts_idx] += 1
        
        # Try removing blocks in random order
        indices_list = list(best_indices)
        rng = random.Random(int(time.time() * 1000))
        rng.shuffle(indices_list)
        
        for b_idx in indices_list:
            if b_idx not in best_indices:
                continue
            
            # Check if removing this block would uncover any t-set
            # i.e., for all ts in block, cover_count[ts] > 1
            can_remove = True
            for ts_idx in block_cover_indices_arr[b_idx]:
                if cover_count[ts_idx] <= 1:
                    can_remove = False
                    break
            
            if can_remove:
                # Remove block
                best_indices.remove(b_idx)
                for ts_idx in block_cover_indices_arr[b_idx]:
                    cover_count[ts_idx] -= 1
                improved = True
                best_size -= 1
    
    # Reconstruct solution from indices
    final_solution = [all_blocks[i] for i in best_indices]
    
    return final_solution

def _greedy_cover_with_indices(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed=42, randomize=False, t_set_to_blocks=None):
    rng = random.Random(rng_seed)
    
    uncovered = set(range(num_t_sets))
    selected_indices = []
    
    # Greedy Loop
    while uncovered:
        if time.time() > deadline - 2.0:
            return None, None
        
        target_tset_idx = rng.choice(list(uncovered)) if randomize else min(uncovered)
        
        candidate_block_indices = t_set_to_blocks[target_tset_idx]
        
        best_block_idx = -1
        max_gain = 0
        
        for b_idx in candidate_block_indices:
            gain = 0
            for ts_idx in block_cover_indices[b_idx]:
                if ts_idx in uncovered:
                    gain += 1
            
            if gain > max_gain:
                max_gain = gain
                best_block_idx = b_idx
        
        if best_block_idx == -1:
            break
        
        selected_indices.append(best_block_idx)
        
        for ts_idx in block_cover_indices[best_block_idx]:
            if ts_idx in uncovered:
                uncovered.remove(ts_idx)
        
        if not uncovered:
            break
    
    selected_blocks = [all_blocks[i] for i in selected_indices]
    return selected_blocks, selected_indices

def _greedy_cover(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed=42, randomize=False, t_set_to_blocks=None):
    # Kept for compatibility, but we use _greedy_cover_with_indices now
    blocks, indices = _greedy_cover_with_indices(v, k, t, t_set_index, all_blocks, block_cover_indices, num_t_sets, deadline, rng_seed, randomize, t_set_to_blocks)
    return blocks

# EVOLVE-BLOCK-END
