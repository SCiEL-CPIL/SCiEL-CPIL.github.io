    import itertools
    import random
    import time
    import numpy as np

    def construct(v, k, t):
        start_time = time.time()
        deadline = start_time + 145.0
        
        # Precompute all t-subsets
        all_t_subsets = list(itertools.combinations(range(v), t))
        num_t = len(all_t_subsets)
        t_sub_to_idx = {sub: i for i, sub in enumerate(all_t_subsets)}
        
        # Precompute all possible k-blocks
        all_blocks = list(itertools.combinations(range(v), k))
        num_blocks = len(all_blocks)
        
        # Precompute coverage for each block
        # block_tsubs[i] is a list of t-subset indices covered by block i
        block_tsubs = []
        for block in all_blocks:
            tsubs = []
            for sub in itertools.combinations(block, t):
                tsubs.append(t_sub_to_idx[sub])
            block_tsubs.append(tsubs)
            
        # Map block tuple to index
        block_to_idx = {b: i for i, b in enumerate(all_blocks)}
        
        best_blocks = None
        best_count = float('inf')
        
        # Helper: Remove redundant blocks
        def remove_redundant(blocks):
            if not blocks:
                return blocks
            
            # Map block tuple to index in the current list for quick lookup?
            # Better: Maintain coverage counts.
            coverage_count = np.zeros(num_t, dtype=np.int32)
            block_indices = []
            for b in blocks:
                idx = block_to_idx[b]
                block_indices.append(idx)
                for t_idx in block_tsubs[idx]:
                    coverage_count[t_idx] += 1
                    
            changed = True
            while changed:
                changed = False
                if time.time() > deadline - 5.0:
                    break
                
                # Find redundant blocks
                redundant = []
                for i, b_idx in enumerate(block_indices):
                    is_essential = False
                    for t_idx in block_tsubs[b_idx]:
                        if coverage_count[t_idx] == 1:
                            is_essential = True
                            break
                    if not is_essential:
                        redundant.append(i)
                        
                if redundant:
                    # Remove one
                    rem_idx = random.choice(redundant)
                    rem_b_idx = block_indices[rem_idx]
                    # Decrement coverage
                    for t_idx in block_tsubs[rem_b_idx]:
                        coverage_count[t_idx] -= 1
                    block_indices.pop(rem_idx)
                    changed = True
                    
            return [all_blocks[i] for i in block_indices]

        # Helper: Greedy Construction
        def greedy_construct(seed):
            random.seed(seed)
            uncovered = np.ones(num_t, dtype=bool)
            blocks = []
            
            while uncovered.any():
                # Pick random uncovered t-subset
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                seed_idx = random.choice(uncovered_indices)
                seed_subset = all_t_subsets[seed_idx]
                block = list(seed_subset)
                
                while len(block) < k:
                    best_x = -1
                    best_gain = -1
                    tied = []
                    
                    block_set = set(block)
                    # Precompute t-1 subsets of current block
                    t_minus_1_subsets = list(itertools.combinations(block, t - 1))
                    
                    for x in range(v):
                        if x in block_set:
                            continue
                        gain = 0
                        for rest in t_minus_1_subsets:
                            T = tuple(sorted(rest + (x,)))
                            if T in t_sub_to_idx:
                                idx = t_sub_to_idx[T]
                                if uncovered[idx]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            tied = [x]
                        elif gain == best_gain:
                            tied.append(x)
                            
                    if best_x == -1:
                        break
                    if len(tied) > 1:
                        best_x = random.choice(tied)
                    block.append(best_x)
                    
                block.sort()
                blocks.append(block)
                
                # Update uncovered
                b_idx = block_to_idx[tuple(block)]
                for t_idx in block_tsubs[b_idx]:
                    uncovered[t_idx] = False
                    
            return blocks

        # Helper: Local Search (Swap)
        def local_search(blocks):
            # Convert to indices
            block_indices = [block_to_idx[b] for b in blocks]
            coverage_count = np.zeros(num_t, dtype=np.int32)
            for b_idx in block_indices:
                for t_idx in block_tsubs[b_idx]:
                    coverage_count[t_idx] += 1
            
            improved = True
            while improved:
                improved = False
                if time.time() > deadline - 5.0:
                    break
                    
                # First, remove any redundant
                new_block_indices = remove_redundant_from_indices(block_indices, coverage_count)
                if len(new_block_indices) < len(block_indices):
                    block_indices = new_block_indices
                    # Recalculate coverage for remaining
                    coverage_count = np.zeros(num_t, dtype=np.int32)
                    for b_idx in block_indices:
                        for t_idx in block_tsubs[b_idx]:
                            coverage_count[t_idx] += 1
                    improved = True
                    continue
                
                # Try swaps
                # Pick a random block to try to replace
                i = random.randint(0, len(block_indices) - 1)
                b_i = block_indices[i]
                
                # Find t-subsets uniquely covered by b_i
                unique_t_idxs = []
                for t_idx in block_tsubs[b_i]:
                    if coverage_count[t_idx] == 1:
                        unique_t_idxs.append(t_idx)
                        
                if not unique_t_idxs:
                    # Redundant, should have been removed
                    continue
                    
                # Find candidate blocks that cover all unique_t_idxs
                # And are not already in the list
                current_set = set(block_indices)
                candidates = []
                for b_j in range(num_blocks):
                    if b_j in current_set:
                        continue
                    # Check if b_j covers all unique_t_idxs
                    # Optimization: Precompute a bitmask or just check intersection?
                    # For speed, check if set(unique_t_idxs) is subset of set(block_tsubs[b_j])
                    # This is slow.
                    # Let's just pick a random few candidates
                    pass 
                
                # This swap search is too slow in Python if done naively.
                # Let's stick to just redundancy removal and multiple greedy restarts.
                
            return [all_blocks[i] for i in block_indices]

        # ... main loop ...
    