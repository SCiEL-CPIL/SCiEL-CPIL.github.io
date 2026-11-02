import itertools
import random
import time
import numpy as np

def construct(v, k, t):
    """
    Finds a covering design C(v, k, t) using Greedy Construction followed by Local Search (Redundancy Removal).
    """
    # Precompute all t-subsets and map to indices
    all_t_subsets = list(itertools.combinations(range(v), t))
    n_t_subsets = len(all_t_subsets)
    t_sub_index = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Precompute all k-subsets and their covered t-subset indices
    all_k_subsets = list(itertools.combinations(range(v), k))
    n_k_subsets = len(all_k_subsets)
    
    # cover_matrix: shape (n_k_subsets, C(k,t))
    # Each entry is the index of the t-subset covered by that block at that position
    c_k_t = len(list(itertools.combinations(range(k), t)))
    cover_matrix = np.zeros((n_k_subsets, c_k_t), dtype=np.int32)
    
    for i, block in enumerate(all_k_subsets):
        for j, ts in enumerate(itertools.combinations(block, t)):
            cover_matrix[i, j] = t_sub_index[ts]
            
    best_blocks = []
    best_count = float('inf')
    
    start_time = time.time()
    timeout = 115
    
    # Number of restarts
    # Each restart involves a greedy phase and a removal phase.
    # Greedy phase: ~80 steps. Each step scans n_k_subsets * c_k_t? 
    # We can optimize the greedy phase by using the cover_matrix.
    # gains = np.sum(uncovered[cover_matrix], axis=1) is fast.
    
    for restart in range(200):
        if time.time() - start_time > timeout:
            break
        
        random.seed(restart)
        
        # --- Phase 1: Greedy Construction ---
        uncovered = np.ones(n_t_subsets, dtype=bool)
        blocks_found = []
        
        while np.any(uncovered):
            # Calculate gain for all blocks
            # uncovered[cover_matrix] creates an array of shape (n_k_subsets, c_k_t)
            # Summing along axis 1 gives the count of uncovered triples covered by each block
            gains = np.sum(uncovered[cover_matrix], axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break # Should not happen if uncovered is not empty and we have all blocks
            
            # Pick randomly from blocks with max gain to diversify
            candidates = np.where(gains == max_gain)[0]
            chosen_idx = candidates[random.randint(0, len(candidates) - 1)]
            
            blocks_found.append(chosen_idx)
            
            # Update uncovered
            uncovered[cover_matrix[chosen_idx]] = False
            
        # --- Phase 2: Local Search (Redundancy Removal) ---
        # We have a list of block indices.
        # We want to remove any block that is not "essential".
        # A block is essential if it is the *only* block covering at least one of its triples.
        
        # Initialize cover counts
        # We can use a dictionary or an array. Array is faster.
        cover_count = np.zeros(n_t_subsets, dtype=np.int32)
        
        for idx in blocks_found:
            cover_count[cover_matrix[idx]] += 1
            
        # Iteratively remove redundant blocks
        # We need to be careful with the order. Removing one block might make another redundant.
        # We loop until no more removals are possible.
        changed = True
        # To avoid bias, we can shuffle the blocks before checking?
        # Or just iterate. The final set might depend on order, so we rely on restarts.
        
        # Convert to list for easier modification
        current_blocks = list(blocks_found)
        
        while changed:
            changed = False
            # Shuffle to randomize removal order in this iteration?
            # random.shuffle(current_blocks)
            
            i = 0
            while i < len(current_blocks):
                idx = current_blocks[i]
                # Check if all triples in this block are covered by >1 block
                # cover_matrix[idx] gives the indices of the triples
                triple_indices = cover_matrix[idx]
                
                # If min(cover_count[triple_indices]) > 1, then this block is redundant
                if np.min(cover_count[triple_indices]) > 1:
                    # Remove it
                    # Decrement counts
                    cover_count[triple_indices] -= 1
                    # Remove from list
                    current_blocks.pop(i)
                    changed = True
                    # Do not increment i, check the next element (which is now at i)
                else:
                    i += 1
                    
        # Update best
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            # Convert indices back to tuples/lists
            best_blocks = [all_k_subsets[idx] for idx in current_blocks]
            
            # If we reached the lower bound or known best, we might stop?
            # Known best is 76.
            if best_count <= 76:
                # We can try to continue to see if we get 75, but it's rare.
                # Given time constraints, we might break if we hit 76 multiple times?
                # No, just keep going until timeout.
                pass

    return best_blocks
