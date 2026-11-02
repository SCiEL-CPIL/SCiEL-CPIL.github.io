import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit

    # 1. Precomputation
    points = list(range(v))
    triples = list(itertools.combinations(points, t))
    n_triples = len(triples)
    triple_to_idx = {triple: i for i, triple in enumerate(triples)}
    
    blocks = list(itertools.combinations(points, k))
    n_blocks = len(blocks)
    
    # C(k, t)
    c_k_t = 1
    for i in range(1, t+1):
        c_k_t = c_k_t * (k - i + 1) // i
        
    # block_triples[i] contains the indices of the t-subsets covered by blocks[i]
    # Shape: (n_blocks, c_k_t)
    block_triples = np.zeros((n_blocks, c_k_t), dtype=np.int32)
    
    # Also compute frequency of each triple to use for weighting
    triple_freq = np.zeros(n_triples, dtype=np.int32)
    
    for i, block in enumerate(blocks):
        idx = 0
        for triple in itertools.combinations(block, t):
            t_idx = triple_to_idx[triple]
            block_triples[i, idx] = t_idx
            triple_freq[t_idx] += 1
            idx += 1
            
    # Weights: 1 / frequency. Higher weight for rarer triples.
    # Avoid division by zero (though freq > 0 for all triples in a covering design context)
    weights = 1.0 / np.maximum(triple_freq, 1)
    
    best_solution = None
    best_size = float('inf')
    
    # Number of restarts to attempt
    # We will run until deadline is approaching
    restarts = 0
    while time.time() < deadline - 5.0:
        restarts += 1
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        
        # Weighted Greedy
        while np.any(uncovered):
            # Calculate gains
            # weighted_uncovered[j] = weights[j] if uncovered[j] else 0
            weighted_uncovered = np.where(uncovered, weights, 0.0)
            
            # gains[i] = sum(weighted_uncovered[block_triples[i]])
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == 0:
                break
            
            selected_blocks.append(best_idx)
            uncovered[block_triples[best_idx]] = False
            
        # Pruning Phase
        # selected_blocks is a list of block indices
        if len(selected_blocks) < best_size:
            # Check if we can prune
            # cover_count[j] = number of blocks in selected_blocks covering triple j
            cover_count = np.zeros(n_triples, dtype=np.int32)
            for b_idx in selected_blocks:
                cover_count[block_triples[b_idx]] += 1
                
            # Try to remove blocks
            # A block is redundant if all its triples are covered by at least one other block
            # i.e., for all j in block, cover_count[j] > 1
            # We iterate and remove if possible
            changed = True
            current_selected = list(selected_blocks)
            while changed:
                changed = False
                for i, b_idx in enumerate(current_selected):
                    # Check if block b_idx is redundant
                    # Get the triples covered by this block
                    b_triples = block_triples[b_idx]
                    # If min cover_count for these triples is > 1, it's redundant
                    if np.min(cover_count[b_triples]) > 1:
                        # Remove it
                        # Decrement cover counts
                        cover_count[b_triples] -= 1
                        current_selected.pop(i)
                        changed = True
                        # Restart the loop to check new redundancies
                        break
                        
            if len(current_selected) < best_size:
                best_size = len(current_selected)
                # Convert indices back to blocks
                best_solution = [list(blocks[i]) for i in current_selected]
                
        # Early exit if we hit the lower bound (69)
        if best_size <= 69:
            break

    if best_solution is None:
        # Fallback: return a valid covering, even if not optimal
        # Just return the first greedy result without pruning if pruning failed or time ran out
        # This shouldn't happen if we completed at least one loop
        uncovered = np.ones(n_triples, dtype=bool)
        selected_blocks = []
        while np.any(uncovered):
            weighted_uncovered = np.where(uncovered, weights, 0.0)
            gains = np.sum(weighted_uncovered[block_triples], axis=1)
            best_idx = np.argmax(gains)
            if gains[best_idx] == 0:
                break
            selected_blocks.append(best_idx)
            uncovered[block_triples[best_idx]] = False
        best_solution = [list(blocks[i]) for i in selected_blocks]
        
    return best_solution
