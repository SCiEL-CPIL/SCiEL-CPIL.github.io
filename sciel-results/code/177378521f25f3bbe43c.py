import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Precompute t-subset -> block indices
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    # Helper: Check if a set of block indices forms a valid cover
    # We assume coverage is maintained externally for speed in local search.
    
    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts
    max_restarts = 20 # Reduced restarts, more time per restart for better local search
    
    for seed in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
        random.seed(seed)
        
        # --- Phase 1: Greedy Construction ---
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        uncovered = set(range(num_t_subsets))
        current_blocks = []
        
        while uncovered:
            # Find the uncovered t-subset with the fewest candidate blocks (hardest to cover)
            # If ties, pick randomly or first.
            # Optimization: Instead of checking all, we can just pick a random one from uncovered
            # and its candidates. "Hardest first" is better.
            
            # To avoid O(N_uncovered * N_candidates) which is slow, 
            # we can just pick a random uncovered t_idx.
            # Let's try "Hardest First" by sampling a few.
            sample_size = min(10, len(uncovered))
            candidates_t = random.sample(list(uncovered), sample_size)
            
            best_t_idx = None
            min_candidate_count = float('inf')
            for t_idx in candidates_t:
                count = len(t_to_blocks[t_idx])
                if count < min_candidate_count:
                    min_candidate_count = count
                    best_t_idx = t_idx
            
            # Get blocks covering best_t_idx
            block_candidates = t_to_blocks[best_t_idx]
            
            best_b_idx = -1
            best_gain = -1
            
            for b_idx in block_candidates:
                gain = 0
                for t2_idx in block_t_indices[b_idx]:
                    if t2_idx in uncovered:
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            current_blocks.append(best_b_idx)
            for t2_idx in block_t_indices[best_b_idx]:
                coverage[t2_idx] += 1
                if coverage[t2_idx] == 1:
                    uncovered.discard(t2_idx)
                    
        # --- Phase 2: Local Search (Deletion and Swap) ---
        
        def try_remove_redundant(blocks, coverage):
            """
            Attempts to remove redundant blocks.
            Returns (new_blocks, new_coverage, removed_count)
            """
            changed = True
            removed = 0
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b_idx = blocks[i]
                    is_redundant = True
                    for t_idx in block_t_indices[b_idx]:
                        if coverage[t_idx] <= 1:
                            is_redundant = False
                            break
                    
                    if is_redundant:
                        # Remove block
                        blocks.pop(i)
                        for t_idx in block_t_indices[b_idx]:
                            coverage[t_idx] -= 1
                        changed = True
                        removed += 1
                        break # Restart loop to check for new redundancies
            return blocks, coverage, removed

        current_blocks, coverage, _ = try_remove_redundant(current_blocks, coverage)
        
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]

        # Local Search: Swaps
        # We try to swap blocks to enable further deletions.
        max_swaps = 5000
        swaps_attempted = 0
        no_improvement_counter = 0
        
        while swaps_attempted < max_swaps:
            if time.time() - start_time > time_limit:
                break
            
            if not current_blocks:
                break
            
            # Pick a random block to replace
            i = random.randint(0, len(current_blocks) - 1)
            old_b_idx = current_blocks[i]
            
            # Pick a random new block
            new_b_idx = random.randint(0, num_k_subsets - 1)
            
            if old_b_idx == new_b_idx:
                continue
                
            # Check if swap is valid
            old_t_set = block_t_indices[old_b_idx]
            new_t_set = block_t_indices[new_b_idx]
            
            # Optimization: Use sets for fast intersection/difference? 
            # Or just iterate. 
            # A swap is valid if for all t in old_t_set \ new_t_set, coverage[t] > 1.
            
            valid = True
            # Create a set for new_t_set for O(1) lookup
            new_t_set_as_set = set(new_t_set)
            
            for t_idx in old_t_set:
                if t_idx not in new_t_set_as_set:
                    if coverage[t_idx] <= 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Perform swap
            for t_idx in old_t_set:
                coverage[t_idx] -= 1
            for t_idx in new_t_set:
                coverage[t_idx] += 1
            
            current_blocks[i] = new_b_idx
            
            # Try to remove redundant blocks
            old_size = len(current_blocks)
            current_blocks, coverage, removed = try_remove_redundant(current_blocks, coverage)
            new_size = len(current_blocks)
            
            if new_size < old_size:
                # Improvement!
                if new_size < best_count:
                    best_count = new_size
                    best_blocks = current_blocks[:]
                no_improvement_counter = 0
            else:
                # No immediate improvement. 
                # We can keep the swap to explore new states, or revert.
                # Reverting is safer to maintain a good cover, but might get stuck.
                # Let's revert if no improvement to keep the cover "tight".
                # However, sometimes a non-improving swap leads to a later improvement.
                # Given time limits, let's be greedy: revert.
                # Revert swap
                for t_idx in new_t_set:
                    coverage[t_idx] -= 1
                for t_idx in old_t_set:
                    coverage[t_idx] += 1
                current_blocks[i] = old_b_idx
                
                no_improvement_counter += 1
                if no_improvement_counter > 100:
                    # Stuck, break out of swap loop for this restart
                    break

            swaps_attempted += 1

    # Convert indices back to lists of integers
    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
