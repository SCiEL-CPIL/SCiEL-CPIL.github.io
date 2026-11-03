import itertools
import time
import random
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for the 120s limit
    
    # Precomputation
    points = range(v)
    t_subsets = list(itertools.combinations(points, t))
    num_t_subsets = len(t_subsets)
    
    # Map each t-subset to an index
    t_subset_to_idx = {s: i for i, s in enumerate(t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_blocks = list(itertools.combinations(points, k))
    
    # For each block, find the indices of the t-subsets it covers
    # block_t_indices[i] is a list of indices into t_subsets
    block_t_indices = []
    # Also, for each t-subset, store how many blocks contain it (for weighting)
    t_subset_block_count = [0] * num_t_subsets
    
    for block in all_blocks:
        indices = []
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_to_idx[t_sub]
            indices.append(idx)
            t_subset_block_count[idx] += 1
        block_t_indices.append(indices)
        
    # Weights for t-subsets: 1 / count. Rare subsets are heavier.
    # This helps the greedy algorithm prioritize hard-to-cover subsets.
    weights = [1.0 / count if count > 0 else 0.0 for count in t_subset_block_count]
    
    best_solution = None
    best_score = float('inf')
    
    # Number of restarts depends on time available.
    # Let's estimate time per greedy run.
    # Greedy loop: ~150 iterations.
    # Each iteration: find best block among ~50000.
    # Naive scan: 50000 * 35 ops = 1.75M ops.
    # 150 * 1.75M = 262M ops.
    # In Python, this might take ~5-10 seconds per run.
    # So we can probably do 5-10 restarts.
    
    num_restarts = 10
    
    for r in range(num_restarts):
        if time.time() > deadline - 15.0:
            break
            
        # Greedy Construction
        cover_count = [0] * num_t_subsets
        current_blocks = []
        used_blocks = set()
        
        # To allow variation, we can slightly randomize weights or tie-breaking
        # Here we just rely on the deterministic weighted greedy, but maybe shuffle blocks?
        # No, the order of evaluation doesn't matter if we take the max.
        # To get different solutions, we can randomize the weights slightly.
        randomized_weights = [w * random.uniform(0.8, 1.2) for w in weights]
        
        while any(c == 0 for c in cover_count):
            best_block_idx = -1
            best_gain = -1
            
            # Optimization: Only consider blocks that have at least one uncovered subset?
            # Hard to check efficiently without scanning.
            # Just scan all.
            
            # We can speed this up by maintaining a list of "active" blocks?
            # But a block becomes inactive only if all its subsets are covered.
            
            # Let's just scan.
            # To make it faster, we can pre-sort blocks? No, gain changes.
            
            for b_idx, t_indices in enumerate(block_t_indices):
                if b_idx in used_blocks:
                    continue
                
                gain = 0.0
                for idx in t_indices:
                    if cover_count[idx] == 0:
                        gain += randomized_weights[idx]
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = b_idx
            
            if best_block_idx == -1:
                # Should not happen if problem is solvable
                break
                
            current_blocks.append(all_blocks[best_block_idx])
            used_blocks.add(best_block_idx)
            
            for idx in block_t_indices[best_block_idx]:
                cover_count[idx] += 1
                
        # Post-processing: Remove redundant blocks
        # A block is redundant if for all t-subsets it covers, cover_count > 1
        # Note: cover_count includes the block itself.
        
        changed = True
        while changed:
            changed = False
            new_blocks = []
            for i, block in enumerate(current_blocks):
                # Check if redundant
                redundant = True
                for t_sub in itertools.combinations(block, t):
                    idx = t_subset_to_idx[t_sub]
                    if cover_count[idx] <= 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for t_sub in itertools.combinations(block, t):
                        idx = t_subset_to_idx[t_sub]
                        cover_count[idx] -= 1
                    changed = True
                else:
                    new_blocks.append(block)
            
            current_blocks = new_blocks
            
        score = len(current_blocks)
        if score < best_score:
            best_score = score
            best_solution = current_blocks
            
    return best_solution
