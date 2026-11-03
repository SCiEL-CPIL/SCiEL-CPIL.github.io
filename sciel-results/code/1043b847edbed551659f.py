import itertools
import time
import random
import numpy as np

def construct(v, k, t):
    """
    Construct a covering design C(v, k, t) using a vectorized greedy set cover
    with redundancy removal.
    """
    start_time = time.time()
    deadline = start_time + 115.0 # Safety margin before 120s limit

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
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Fill the matrix. This is the most time-consuming part.
    # Unrolling the inner loop for t=5 for speed.
    for j, lc in enumerate(local_t_combos):
        # lc is a tuple of t indices
        for i in range(n_k):
            block = k_subsets[i]
            # Create the tuple of points
            if t == 5:
                t_combo = (block[lc[0]], block[lc[1]], block[lc[2]], block[lc[3]], block[lc[4]])
            else:
                t_combo = tuple(block[p] for p in lc)
            block_t_indices[i, j] = t_sub_to_idx[t_combo]

    if time.time() > deadline - 10.0:
        # If precomputation took too long, we might not have time for greedy.
        # But we must return something.
        # Let's try to run a very fast greedy or fallback.
        pass

    best_solution = None
    best_size = float('inf')
    
    # Limit restarts to ensure we finish in time.
    num_restarts = 2
    
    for seed in range(num_restarts):
        if time.time() > deadline - 20.0:
            break
            
        sol_indices = _vectorized_greedy_cover(
            n_t, n_k, block_t_indices, n_t_in_k, deadline - 5.0, seed
        )
        
        if sol_indices is None:
            continue
            
        # Post-processing: Remove redundant blocks
        refined_indices = _remove_redundant_blocks(
            sol_indices, block_t_indices, n_t, deadline - 5.0
        )
        
        if len(refined_indices) < best_size:
            best_size = len(refined_indices)
            best_solution = refined_indices

    if best_solution is None:
        # Fallback: Generate a valid covering using a simple heuristic
        return _fallback_construct(v, k, t, deadline)
        
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
        if time.time() > deadline:
            return None # Timeout
            
        # Calculate gain for all blocks
        gains = uncovered[block_t_indices].sum(axis=1)
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        candidates = np.where(gains == max_gain)[0]
        best_idx = rng.choice(candidates)
        
        selected_blocks.append(best_idx)
        uncovered[block_t_indices[best_idx]] = False
        
    return selected_blocks

def _remove_redundant_blocks(selected_indices, block_t_indices, n_t, deadline):
    """
    Iteratively remove blocks that do not cover any t-subset uniquely.
    """
    if not selected_indices:
        return selected_indices
        
    coverage = np.zeros(n_t, dtype=np.int32)
    for b_idx in selected_indices:
        coverage[block_t_indices[b_idx]] += 1
        
    remaining = list(selected_indices)
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        
        for i in range(len(remaining) - 1, -1, -1):
            b_idx = remaining[i]
            ts_indices = block_t_indices[b_idx]
            
            if np.all(coverage[ts_indices] > 1):
                coverage[ts_indices] -= 1
                remaining.pop(i)
                improved = True
                break
                
    return remaining

def _fallback_construct(v, k, t, deadline):
    """
    A simple, guaranteed-to-finish (if time permits) heuristic to generate a valid covering.
    """
    uncovered = set(itertools.combinations(range(v), t))
    blocks = []
    
    while uncovered:
        if time.time() > deadline - 1.0:
            # If we are about to time out, we must return a valid covering.
            # This loop might not finish. 
            # To guarantee validity, we can add blocks for each remaining uncovered t-set.
            for t_set in uncovered:
                remaining_points = [p for p in range(v) if p not in t_set]
                extra = random.sample(remaining_points, k - t)
                blocks.append(tuple(sorted(list(t_set) + extra)))
            uncovered = set()
            break
            
        # Pick a random uncovered t-set
        t_set = random.choice(list(uncovered))
        # Generate a random block containing this t-set
        remaining_points = [p for p in range(v) if p not in t_set]
        extra = random.sample(remaining_points, k - t)
        block = tuple(sorted(list(t_set) + extra))
        
        blocks.append(block)
        
        # Remove all t-sets covered by this block
        covered_t_sets = set(itertools.combinations(block, t))
        uncovered -= covered_t_sets
        
    return [list(b) for b in blocks]


# Helper functions are defined in the previous edit block if we replace the whole file structure, 
# but here we are doing SEARCH/REPLACE on the existing file.
# The previous SEARCH/REPLACE replaced the `construct` function and added the new helpers.
# We must ensure we don't duplicate them.
# The original file had `_vectorized_greedy_cover`, `_remove_redundant_blocks`, and `_local_search_refine` at the end.
# My first edit replaced `construct` and *added* the new versions of `_vectorized_greedy_cover` and `_remove_redundant_blocks` and `_fallback_construct`.
# This means the old versions are still in the file if I don't remove them.
# I should remove the old helper functions to avoid confusion or name conflicts (though Python uses the last definition).
# Actually, if I define them again, the new definitions will overwrite the old ones.
# But it's cleaner to remove the old ones.

# Since I cannot easily remove the old helpers in a single SEARCH/REPLACE if they are large, 
# and the new definitions in the first edit will shadow them, I will just leave the old ones if they don't interfere.
# Wait, the first edit *replaced* the `construct` function. It did *not* remove the old helpers.
# So the file will have:
# 1. New `construct`
# 2. New `_vectorized_greedy_cover`
# 3. New `_remove_redundant_blocks`
# 4. New `_fallback_construct`
# 5. Old `_vectorized_greedy_cover` (will overwrite new? No, Python defines functions in order of execution. 
#    If the new ones are defined *before* the old ones, the old ones will overwrite the new ones.
#    In my first edit, I inserted the new helpers *immediately after* `construct`.
#    The old helpers are at the *end* of the file.
#    So the old helpers will overwrite the new ones!
#    I must remove the old helpers.

# I will add a second edit to remove the old helper functions.
