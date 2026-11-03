import itertools
import random
import numpy as np
import time

def construct(v, k, t):
    """
    Constructs a covering design C(v, k, t) using Greedy Set Cover 
    and Randomized Local Search with optimized data structures.
    """
    start_time = time.time()
    time_limit = 115.0

    # 1. Precomputation
    # Generate all t-subsets
    all_t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(all_t_subsets)
    t_to_idx = {ts: i for i, ts in enumerate(all_t_subsets)}
    
    # Generate all k-subsets (blocks)
    all_k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(all_k_subsets)
    
    # Precompute block -> t-subset indices mapping
    # block_t_indices[i] is a list of indices of t-subsets covered by all_k_subsets[i]
    block_t_indices = []
    for block in all_k_subsets:
        indices = [t_to_idx[ts] for ts in itertools.combinations(block, t)]
        block_t_indices.append(indices)
        
    # Precompute t-subset -> block indices mapping
    # t_to_blocks[j] is a list of indices of blocks covering all_t_subsets[j]
    t_to_blocks = [[] for _ in range(num_t_subsets)]
    for b_idx, indices in enumerate(block_t_indices):
        for t_idx in indices:
            t_to_blocks[t_idx].append(b_idx)

    best_blocks = None
    best_count = float('inf')
    
    # Multiple restarts with different random seeds for the local search
    max_restarts = 200 # Increase restarts since each is faster
    
    for seed in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(seed)
        
        # --- Phase 1: Greedy Construction ---
        # We want to cover all t-subsets with minimum blocks.
        # Greedy: Pick block with max coverage of uncovered t-subsets.
        
        # To speed up, we can maintain a list of uncovered t-subsets and 
        # for each candidate block, count how many of its t-subsets are uncovered.
        
        # A more efficient greedy approach:
        # 1. Start with all t-subsets uncovered.
        # 2. For each step, find the block that covers the most uncovered t-subsets.
        #    To do this efficiently, we can iterate through uncovered t-subsets, 
        #    and for each, look at blocks covering it. But that's complex.
        
        # Alternative: Randomized Greedy.
        # Pick a random uncovered t-subset. Among blocks covering it, pick one 
        # that covers the most other uncovered t-subsets.
        
        uncovered = set(range(num_t_subsets))
        coverage = np.zeros(num_t_subsets, dtype=np.int32)
        current_blocks = []
        
        while uncovered:
            # Pick a random uncovered t-subset
            t_idx = random.choice(tuple(uncovered))
            
            # Get all blocks that cover this t-subset
            candidates = t_to_blocks[t_idx]
            
            best_b_idx = -1
            best_gain = -1
            
            # Evaluate candidates
            # Optimization: If a candidate covers many uncovered, it's good.
            for b_idx in candidates:
                gain = 0
                # Count how many t-subsets in this block are currently uncovered
                for t2_idx in block_t_indices[b_idx]:
                    if t2_idx in uncovered:
                        gain += 1
                        if gain > best_gain:
                            # Early termination if we can't beat best? 
                            # Max possible gain is len(block_t_indices[b_idx])
                            # If best_gain is already high, we can skip?
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_b_idx = b_idx
            
            # Add the best block
            current_blocks.append(best_b_idx)
            for t2_idx in block_t_indices[best_b_idx]:
                coverage[t2_idx] += 1
                if coverage[t2_idx] == 1:
                    uncovered.discard(t2_idx)
                    
        # --- Phase 2: Redundancy Removal (Cleanup) ---
        def remove_redundant(blocks, coverage):
            """
            Removes redundant blocks from the list.
            Returns the new list of block indices and updated coverage.
            """
            changed = True
            while changed:
                changed = False
                for i in range(len(blocks)):
                    b_idx = blocks[i]
                    is_redundant = True
                    # Check if all t-subsets in this block are covered by > 1 block
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
                        break # Restart the loop to check for new redundancies
            return blocks, coverage

        current_blocks, coverage = remove_redundant(current_blocks, coverage)
        
        # Update best
        if len(current_blocks) < best_count:
            best_count = len(current_blocks)
            best_blocks = current_blocks[:]

        # --- Phase 3: Randomized Local Search ---
        # We try to swap blocks to find a configuration that allows more removals.
        
        max_swaps = 500
        swaps_attempted = 0
        
        while swaps_attempted < max_swaps:
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random block to replace
            if not current_blocks:
                break
            
            i = random.randint(0, len(current_blocks) - 1)
            old_b_idx = current_blocks[i]
            
            # Pick a random new block
            new_b_idx = random.randint(0, num_k_subsets - 1)
            
            if old_b_idx == new_b_idx:
                continue
                
            # Check if swap is valid
            # A swap is valid if no t-subset coverage drops to 0.
            # Specifically, for any t-subset in old_b_idx that is NOT in new_b_idx,
            # coverage must be > 1.
            
            old_t_set = set(block_t_indices[old_b_idx])
            new_t_set = set(block_t_indices[new_b_idx])
            
            valid = True
            # Check t-subsets that will lose coverage
            for t_idx in old_t_set:
                if t_idx not in new_t_set:
                    if coverage[t_idx] <= 1:
                        valid = False
                        break
            
            if not valid:
                continue
                
            # Perform swap
            # Decrement old
            for t_idx in old_t_set:
                coverage[t_idx] -= 1
            # Increment new
            for t_idx in new_t_set:
                coverage[t_idx] += 1
            
            current_blocks[i] = new_b_idx
            
            # Try to remove redundant blocks
            old_size = len(current_blocks)
            current_blocks, coverage = remove_redundant(current_blocks, coverage)
            new_size = len(current_blocks)
            
            # Acceptance criterion
            # If size decreased, accept.
            # If size same, accept with some probability to explore.
            # If size increased (shouldn't happen if remove_redundant is perfect, but maybe 
            # the swap itself didn't allow removal and we just swapped), we should revert?
            
            if new_size > old_size:
                # Revert
                for t_idx in new_t_set:
                    coverage[t_idx] -= 1
                for t_idx in old_t_set:
                    coverage[t_idx] += 1
                current_blocks[i] = old_b_idx
            elif new_size == old_size:
                # Keep with 50% probability? Or always keep to change state?
                # Let's always keep if size <= old_size.
                pass
            else:
                # Size decreased, keep
                pass

            swaps_attempted += 1

            # Update best
            if len(current_blocks) < best_count:
                best_count = len(current_blocks)
                best_blocks = current_blocks[:]
                
    # Convert indices back to lists of integers
    final_blocks = [list(all_k_subsets[idx]) for idx in best_blocks]
    return final_blocks
