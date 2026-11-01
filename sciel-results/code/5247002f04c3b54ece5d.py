import itertools
import random
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin
    
    # Precompute all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    subset_to_idx = {s: i for i, s in enumerate(all_t_subsets)}
    num_subsets = len(all_t_subsets)
    
    # We will track coverage counts
    # cover_count[i] = number of blocks covering subset i
    
    def greedy_construct(random_seed=0):
        random.seed(random_seed)
        cover_count = [0] * num_subsets
        uncovered = set(range(num_subsets))
        blocks = []
        
        # Pre-compute for each t-subset, the list of (k-t) points that can be added to form a block
        # Actually, it's better to generate candidate blocks on the fly based on an uncovered subset
        
        while uncovered:
            # Pick a random uncovered subset, or the one with fewest candidates
            # For speed, let's just pick a random one from the uncovered set
            # To be more strategic, we can pick the one that is in the fewest potential blocks?
            # All t-subsets have the same number of potential blocks: C(v-t, k-t)
            # So random is fine, or we can use the first one.
            
            if not uncovered:
                break
            
            # Pick a target subset to cover
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            target_set = set(target_subset)
            
            # Generate candidate blocks containing target_subset
            # We need to choose k-t points from the remaining v-t points
            remaining_points = [p for p in range(v) if p not in target_set]
            
            best_block = None
            best_gain = -1
            
            # Iterate through combinations of k-t points from remaining
            # C(14, 3) = 364 combinations
            for combo in itertools.combinations(remaining_points, k - t):
                block_tuple = tuple(sorted(target_subset + combo))
                
                # Calculate gain: how many currently uncovered subsets does this block cover?
                gain = 0
                # Iterate through all t-subsets of this block
                # There are C(k, t) = C(8, 5) = 56 such subsets
                for sub in itertools.combinations(block_tuple, t):
                    idx = subset_to_idx[sub]
                    if idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tuple
            
            # Add the best block
            if best_block is not None:
                blocks.append(best_block)
                # Update coverage
                for sub in itertools.combinations(best_block, t):
                    idx = subset_to_idx[sub]
                    cover_count[idx] += 1
                    if cover_count[idx] == 1:
                        uncovered.discard(idx)
        
        return blocks, cover_count, uncovered

    def local_search_reduce(blocks, cover_count, uncovered):
        # Try to remove redundant blocks
        improved = True
        while improved:
            improved = False
            # Shuffle blocks to avoid bias
            random.shuffle(blocks)
            for i in range(len(blocks)):
                block = blocks[i]
                # Check if removing this block uncovers any subset
                redundant = True
                for sub in itertools.combinations(block, t):
                    idx = subset_to_idx[sub]
                    if cover_count[idx] == 1:
                        redundant = False
                        break
                
                if redundant:
                    # Remove it
                    for sub in itertools.combinations(block, t):
                        idx = subset_to_idx[sub]
                        cover_count[idx] -= 1
                        if cover_count[idx] == 0:
                            uncovered.add(idx)
                    blocks.pop(i)
                    improved = True
                    break # Restart the loop to check again from beginning
        return blocks, cover_count, uncovered

    def perturb_and_recover(blocks, cover_count, uncovered):
        # Randomly remove a block and re-cover
        if not blocks:
            return blocks, cover_count, uncovered
        
        # Pick a random block to remove
        idx = random.randint(0, len(blocks) - 1)
        block_to_remove = blocks[idx]
        
        # Remove it
        for sub in itertools.combinations(block_to_remove, t):
            s_idx = subset_to_idx[sub]
            cover_count[s_idx] -= 1
            if cover_count[s_idx] == 0:
                uncovered.add(s_idx)
        blocks.pop(idx)
        
        # Now, use greedy to cover the newly uncovered subsets
        # We can reuse the greedy logic but only for uncovered ones
        # To save time, we just run a few steps of greedy
        # Actually, just calling greedy_construct on the current state is hard because we need to preserve existing blocks.
        # So we implement a "complete" function.
        
        while uncovered:
            target_idx = random.choice(list(uncovered))
            target_subset = all_t_subsets[target_idx]
            target_set = set(target_subset)
            remaining_points = [p for p in range(v) if p not in target_set]
            
            best_block = None
            best_gain = -1
            
            for combo in itertools.combinations(remaining_points, k - t):
                block_tuple = tuple(sorted(target_subset + combo))
                gain = 0
                for sub in itertools.combinations(block_tuple, t):
                    s_idx = subset_to_idx[sub]
                    if s_idx in uncovered:
                        gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_block = block_tuple
            
            if best_block is not None:
                blocks.append(best_block)
                for sub in itertools.combinations(best_block, t):
                    s_idx = subset_to_idx[sub]
                    cover_count[s_idx] += 1
                    if cover_count[s_idx] == 1:
                        uncovered.discard(s_idx)
                        
        return blocks, cover_count, uncovered

    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # 1. Initial Greedy
        blocks, cover_count, uncovered = greedy_construct(random_seed=i)
        
        # 2. Local Search Reduction
        blocks, cover_count, uncovered = local_search_reduce(blocks, cover_count, uncovered)
        
        if len(blocks) < best_len:
            best_len = len(blocks)
            best_blocks = blocks.copy()
            
        # 3. Perturbation and Recovery (Iterated Local Search)
        # Do a few iterations of perturb/recover/reduce
        for _ in range(5):
            if time.time() > deadline:
                break
            blocks, cover_count, uncovered = perturb_and_recover(blocks, cover_count, uncovered)
            blocks, cover_count, uncovered = local_search_reduce(blocks, cover_count, uncovered)
            
            if len(blocks) < best_len:
                best_len = len(blocks)
                best_blocks = blocks.copy()

    if best_blocks is None:
        # Fallback: simple greedy
        blocks, _, _ = greedy_construct(0)
        return blocks

    return best_blocks
