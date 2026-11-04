import itertools
import numpy as np
import time

def remove_redundant(blocks, all_subsets, subset_to_idx, v, k, t):
    """
    Remove blocks that are redundant (their coverage is fully covered by other blocks).
    """
    if not blocks:
        return []
    
    # Map each block to its index in the list
    # We need to check for each block B if all its t-subsets are covered by blocks != B
    
    # Build a set of triplets covered by each block
    # block_triplets_set[i] = set of triplet indices covered by blocks[i]
    block_triplets_sets = []
    for b in blocks:
        trip = set()
        for comb in itertools.combinations(b, t):
            trip.add(subset_to_idx[comb])
        block_triplets_sets.append(trip)
    
    n = len(blocks)
    keep = [True] * n
    
    # Iterate to fixpoint
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if not keep[i]:
                continue
            # Check if all triplets in block i are covered by other kept blocks
            is_redundant = True
            for triplet_idx in block_triplets_sets[i]:
                # Check if any other kept block covers this triplet
                covered_by_other = False
                for j in range(n):
                    if j == i or not keep[j]:
                        continue
                    if triplet_idx in block_triplets_sets[j]:
                        covered_by_other = True
                        break
                if not covered_by_other:
                    is_redundant = False
                    break
            
            if is_redundant:
                keep[i] = False
                changed = True
                
    return [blocks[i] for i in range(n) if keep[i]]

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 140  # 10s margin for safety
    
    # 1. Precomputation
    # Generate all t-subsets and map to index
    all_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    n_subsets = len(all_subsets)
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(range(v), k))
    n_blocks = len(all_blocks)
    
    # Precompute triplet indices for each block
    # Using a list of lists first, then convert to numpy array for speed
    block_triplets_list = []
    for block in all_blocks:
        trip_indices = []
        for comb in itertools.combinations(block, t):
            trip_indices.append(subset_to_idx[comb])
        block_triplets_list.append(trip_indices)
        
    # Convert to numpy array of shape (n_blocks, C(k, t))
    # All inner lists have same length
    block_triplets = np.array(block_triplets_list, dtype=np.int32)
    
    best_blocks = None
    best_count = float('inf')
    
    # Run multiple greedy restarts with random tie-breaking
    # 100 restarts should take ~20-40 seconds, well within budget
    num_restarts = 100
    
    for seed in range(num_restarts):
        if time.time() > deadline - 10:
            break
            
        np.random.seed(seed)
        uncovered_mask = np.ones(n_subsets, dtype=bool)
        chosen_blocks = []
        
        while True:
            if time.time() > deadline - 5:
                break
                
            # Calculate gains: number of uncovered triplets covered by each block
            # uncovered_mask[all_block_triplets] creates a boolean array of shape (n_blocks, 35)
            # Summing gives the count of True values (uncovered triplets) for each block
            gains = np.sum(uncovered_mask[block_triplets], axis=1)
            
            max_gain = np.max(gains)
            
            if max_gain == 0:
                # All covered? Or stuck?
                break
                
            # Find all blocks with max gain
            candidates = np.where(gains == max_gain)[0]
            
            # Randomly pick one to break ties
            idx = np.random.choice(candidates)
            
            # Add block to solution
            chosen_blocks.append(all_blocks[idx])
            
            # Update uncovered mask
            uncovered_mask[block_triplets[idx]] = False
            
            # Check if all covered
            if not np.any(uncovered_mask):
                break
                
        # Check if we found a complete cover
        if not np.any(uncovered_mask):
            # Remove redundant blocks to minimize count
            final_blocks = remove_redundant(chosen_blocks, all_subsets, subset_to_idx, v, k, t)
            
            if len(final_blocks) < best_count:
                best_count = len(final_blocks)
                best_blocks = final_blocks
                
    # If no valid cover was found, return empty list (should not happen)
    if best_blocks is None:
        return []
        
    return best_blocks
