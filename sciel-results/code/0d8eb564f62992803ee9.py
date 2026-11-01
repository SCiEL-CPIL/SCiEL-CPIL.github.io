import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using Simulated Annealing.
    """
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets (blocks) and the t-subsets they cover
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # For each block, store the indices of the t-subsets it covers
    block_cover_indices = []
    for block in all_k_subsets:
        t_subsets_in_block = list(itertools.combinations(block, t))
        indices = [t_sub_index[ts] for ts in t_subsets_in_block]
        block_cover_indices.append(indices)
        
    # Convert to numpy array for faster access
    cover_matrix = np.array(block_cover_indices, dtype=np.int32)
    
    # State variables
    cover_count = np.zeros(n_t_subsets, dtype=np.int32)
    current_blocks = set()
    uncovered = n_t_subsets
    
    def add_block(idx):
        nonlocal uncovered
        current_blocks.add(idx)
        for tri_idx in cover_matrix[idx]:
            if cover_count[tri_idx] == 0:
                uncovered -= 1
            cover_count[tri_idx] += 1
            
    def remove_block(idx):
        nonlocal uncovered
        if idx not in current_blocks:
            return
        current_blocks.remove(idx)
        for tri_idx in cover_matrix[idx]:
            cover_count[tri_idx] -= 1
            if cover_count[tri_idx] == 0:
                uncovered += 1
                
    def get_cost():
        # Minimize number of blocks, penalize uncovered triples heavily
        return len(current_blocks) + 1000 * uncovered

    # 1. Initialize: Add random blocks until valid (uncovered == 0)
    # We use a simple random addition strategy to get a valid start
    # This is distinct from the greedy "max gain" approach of the seed.
    random.seed(42) # Fixed seed for reproducibility of init, SA will use its own logic
    while uncovered > 0:
        idx = random.randint(0, n_k_subsets - 1)
        if idx not in current_blocks:
            add_block(idx)
            
    best_blocks = list(current_blocks)
    best_cost = get_cost()
    
    # 2. Simulated Annealing
    T = 10.0
    cooling = 0.999
    min_T = 0.01
    start_time = time.time()
    timeout = 115
    
    while T > min_T and (time.time() - start_time) < timeout:
        # Generate a random move
        move_type = random.random()
        
        if move_type < 0.4 and len(current_blocks) > 0:
            # Remove a random block
            if current_blocks:
                idx_to_remove = random.choice(list(current_blocks))
                old_cost = get_cost()
                remove_block(idx_to_remove)
                new_cost = get_cost()
                
                delta = new_cost - old_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(current_blocks)
                else:
                    # Reject move: re-add the block
                    add_block(idx_to_remove)
                    
        elif move_type < 0.7:
            # Add a random block
            idx_to_add = random.randint(0, n_k_subsets - 1)
            if idx_to_add not in current_blocks:
                old_cost = get_cost()
                add_block(idx_to_add)
                new_cost = get_cost()
                
                delta = new_cost - old_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_blocks = list(current_blocks)
                else:
                    # Reject move: remove the block
                    remove_block(idx_to_add)
                    
        else:
            # Swap a block
            if current_blocks:
                idx_in = random.choice(list(current_blocks))
                idx_out = random.randint(0, n_k_subsets - 1)
                if idx_out not in current_blocks and idx_out != idx_in:
                    old_cost = get_cost()
                    remove_block(idx_in)
                    add_block(idx_out)
                    new_cost = get_cost()
                    
                    delta = new_cost - old_cost
                    if delta < 0 or random.random() < np.exp(-delta / T):
                        if new_cost < best_cost:
                            best_cost = new_cost
                            best_blocks = list(current_blocks)
                    else:
                        # Reject move: revert
                        remove_block(idx_out)
                        add_block(idx_in)
                        
        T *= cooling
        
    # Return the best solution found as a list of blocks (tuples of ints)
    return [all_k_subsets[i] for i in best_blocks]
