import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a vectorized greedy set cover
    with redundancy removal and local search refinement.
    """
    start_time = time.time()
    deadline = 110.0 # Safety margin before 120s limit

    # 1. Precomputation
    # All t-subsets of {0, ..., v-1}
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {subset: i for i, subset in enumerate(t_subsets)}
    n_t = len(t_subsets)
    
    # All k-subsets of {0, ..., v-1}
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)
    
    # Local t-combinations within a k-set (positions)
    local_t_combos = list(itertools.combinations(range(k), t))
    n_t_in_k = len(local_t_combos)
    
    # Build dense matrix: block_t_indices[block_idx, local_combo_idx] = global_t_subset_idx
    # This allows vectorized gain calculation: uncovered[block_t_indices[b]].sum()
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # To speed up filling, we iterate over local combos and update columns
    # This is better than iterating over blocks and rows due to cache locality in some cases,
    # but for this size, a simple nested loop is fine.
    for j, lc in enumerate(local_t_combos):
        lc_list = list(lc)
        for i, block in enumerate(k_subsets):
            # Extract the t points from the block at positions lc_list
            # tuple(block[p] for p in lc_list)
            t_combo = tuple(block[p] for p in lc_list)
            block_t_indices[i, j] = t_sub_to_idx[t_combo]
            
    # Build inverted index: t_to_blocks[t_idx] = list of block indices containing this t-subset
    # This is useful for local search or if we want to restrict candidate blocks.
    # For the main greedy loop, we use the dense matrix, so this is optional but good for swaps.
    t_to_blocks = [[] for _ in range(n_t)]
    # Vectorized fill for inverted index is tricky with lists, so we do it in a loop
    # We can optimize by iterating over blocks and their t-indices
    for i in range(n_k):
        for t_idx in block_t_indices[i]:
            t_to_blocks[t_idx].append(i)

    best_solution = None
    best_size = float('inf')
    
    # Multiple strategies/restarts
    # We will run several greedy passes with different random seeds for tie-breaking
    # The vectorized gain calculation is fast, so we can afford many restarts.
    
    num_restarts = 50 # Max restarts, limited by time
    
    for seed in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        sol_indices = _vectorized_greedy_cover(
            n_t, n_k, block_t_indices, n_t_in_k, deadline, seed
        )
        
        if sol_indices is None:
            continue
            
        # Post-processing: Remove redundant blocks
        refined_indices = _remove_redundant_blocks(
            sol_indices, block_t_indices, n_t, deadline
        )
        
        # Optional: Local search swaps (limited by time)
        if time.time() < deadline - 5.0:
            refined_indices = _local_search_refine(
                refined_indices, block_t_indices, n_t, t_to_blocks, k_subsets, deadline
            )
            
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution = refined_indices
            # Early stop if we reach a very low bound? 
            # Theoretical lower bound is around 264. If we get close, we might stop.
            # But let's let it run to find if we can go lower.

    if best_solution is None:
        return []
        
    # Convert indices back to blocks (lists of ints)
    return [list(k_subsets[i]) for i in best_solution]


def _vectorized_greedy_cover(n_t, n_k, block_t_indices, n_t_in_k, deadline, seed):
    """
    Greedy set cover using vectorized gain calculation.
    """
    rng = random.Random(seed)
    uncovered = np.ones(n_t, dtype=np.bool_)
    selected_blocks = []
    
    while uncovered.any():
        if time.time() > deadline - 10.0:
            return None # Timeout, return None to indicate failure for this restart
        
        # Calculate gain for all blocks: number of uncovered t-sets covered by each block
        # uncovered[block_t_indices] creates a 2D boolean array (n_k, n_t_in_k)
        # .sum(axis=1) sums along the t-combinations dimension
        gains = uncovered[block_t_indices].sum(axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            # Should not happen if uncovered.any() is true, unless there's a bug
            break
            
        # Find all blocks with max gain
        # To introduce randomness and diversity, we pick one randomly from the max gain set
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        
        selected_blocks.append(best_idx)
        
        # Update uncovered
        # We can vectorize this update too, but it's a small array (n_t_in_k ~ 56)
        uncovered[block_t_indices[best_idx]] = False
        
    return selected_blocks


def _remove_redundant_blocks(selected_indices, block_t_indices, n_t, deadline):
    """
    Iteratively remove blocks that do not cover any t-subset uniquely.
    """
    if not selected_indices:
        return selected_indices
        
    # Build coverage count
    coverage = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_indices:
        coverage[block_t_indices[b_idx]] += 1
        
    remaining = list(selected_indices)
    improved = True
    while improved:
        if time.time() > deadline - 5.0:
            break
        improved = False
        
        # Iterate in reverse to allow popping
        for i in range(len(remaining) - 1, -1, -1):
            b_idx = remaining[i]
            ts_indices = block_t_indices[b_idx]
            
            # Check if all t-subsets in this block have coverage > 1
            # If yes, the block is redundant
            if np.all(coverage[ts_indices] > 1):
                # Remove block
                coverage[ts_indices] -= 1
                remaining.pop(i)
                improved = True
                break # Restart loop as coverage changed
                
    return remaining


def _local_search_refine(indices, block_t_indices, n_t, t_to_blocks, k_subsets, deadline):
    """
    Simple local search: try to swap a block with another to reduce total count.
    This is heuristic and limited by time.
    """
    if not indices:
        return indices
        
    current_indices = list(indices)
    coverage = np.zeros(n_t, dtype=np.int32)
    for b_idx in current_indices:
        coverage[block_t_indices[b_idx]] += 1
        
    improved = True
    max_iters = 5
    iter_count = 0
    
    while improved and iter_count < max_iters:
        if time.time() > deadline - 2.0:
            break
        improved = False
        iter_count += 1
        
        # Try to remove a block and replace it with a better one?
        # Or just try to find a block that can be removed? (Already done in _remove_redundant)
        # Let's try a simple swap: Pick a block, remove it, see which t-sets become uncovered.
        # Find a new block that covers those t-sets and adds the most other coverage.
        
        # To keep it fast, we only try a few random blocks for removal.
        n_blocks = len(current_indices)
        if n_blocks == 0:
            break
            
        sample_size = min(5, n_blocks)
        candidates_to_remove = random.sample(range(n_blocks), sample_size)
        
        for i in candidates_to_remove:
            if time.time() > deadline - 1.0:
                break
                
            b_idx = current_indices[i]
            ts_indices = block_t_indices[b_idx]
            
            # Temporarily remove
            coverage[ts_indices] -= 1
            uncovered_now = coverage[ts_indices] == 0
            
            if not uncovered_now.any():
                # Safe to remove without replacement (should have been caught by redundant check)
                current_indices.pop(i)
                improved = True
                break
                
            # Find best replacement
            # The replacement must cover all currently uncovered t-sets
            # Let's find candidates from t_to_blocks for the first uncovered t-set
            first_uncovered_idx = np.where(uncovered_now)[0][0]
            global_t_idx = ts_indices[first_uncovered_idx]
            
            potential_replacements = t_to_blocks[global_t_idx]
            
            best_new_block = -1
            best_score = -1
            
            for nb_idx in potential_replacements:
                if nb_idx in current_indices: # Can't add a block already there
                    continue
                    
                nb_ts = block_t_indices[nb_idx]
                
                # Check if it covers all uncovered_now
                # uncovered_now is local to the removed block's t-sets
                # We need to map global indices
                # This check is expensive. Let's approximate:
                # Does nb cover the specific t-sets that became uncovered?
                
                # For each t-set in the removed block that is now uncovered,
                # check if nb covers it.
                
                # Optimization: If nb doesn't cover the first uncovered one, skip (it does by construction of potential_replacements)
                # Check the rest.
                
                valid = True
                for local_idx in range(len(ts_indices)):
                    if uncovered_now[local_idx]:
                        g_idx = ts_indices[local_idx]
                        # Check if g_idx is in nb_ts
                        # Since nb_ts is a small array, we can use a set or search
                        # For speed, we'll use a simple check. 
                        # Actually, we can precompute a set of global indices for nb? No, too slow per call.
                        # Let's just check if the specific global index is in the nb_ts array.
                        if g_idx not in nb_ts:
                            valid = False
                            break
                            
                if not valid:
                    continue
                    
                # Score: number of *other* uncovered t-sets this new block covers
                # We want to maximize coverage of currently uncovered sets (which are 0 in coverage)
                # Note: coverage was decremented for the removed block.
                # So coverage[g_idx] == 0 means it's currently uncovered.
                
                score = 0
                for g_idx in nb_ts:
                    if coverage[g_idx] == 0:
                        score += 1
                        
                if score > best_score:
                    best_score = score
                    best_new_block = nb_idx
                    
            if best_new_block != -1:
                # Perform swap
                # Add new block
                current_indices[i] = best_new_block
                coverage[block_t_indices[best_new_block]] += 1
                improved = True
            else:
                # Undo removal
                coverage[ts_indices] += 1
                
    return current_indices
