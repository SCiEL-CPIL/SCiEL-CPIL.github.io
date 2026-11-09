import itertools
import numpy as np
import time
from math import comb

def construct(v, k, t):
    # Precompute t-subsets and map to indices
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # Precompute k-subsets (blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Precompute coverage matrix: for each block, indices of t-subsets it covers
    t_per_block = comb(k, t)
    coverage_matrix = np.zeros((n_k, t_per_block), dtype=np.int32)
    
    for i, block in enumerate(k_subs):
        subs_in_block = itertools.combinations(block, t)
        indices = [t_sub_idx[sub] for sub in subs_in_block]
        coverage_matrix[i] = indices
        
    # State: which t-subsets are not yet covered
    is_uncovered = np.ones(n_t, dtype=bool)
    
    blocks_found_indices = []
    
    start_time = time.time()
    deadline = start_time + 110.0 # Leave 10s for post-processing and safety
    
    # Greedy Phase
    while np.any(is_uncovered):
        if time.time() > deadline:
            break
            
        # Calculate gains for all blocks
        # taken[i, j] is True if block i covers t-subset j and t-subset j is uncovered
        taken = np.take(is_uncovered, coverage_matrix)
        gains = np.sum(taken, axis=1)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            break
            
        best_idx = np.argmax(gains)
        
        blocks_found_indices.append(best_idx)
        
        is_uncovered[coverage_matrix[best_idx]] = False
        
    # Post-processing: Iterative Reduction
    # We want to remove any block that is redundant.
    # A block is redundant if all t-subsets it covers are covered by other blocks.
    
    def reduce_solution(indices):
        if not indices:
            return indices
            
        # Build reverse index: t_sub_idx -> set of block indices in current solution
        # Using a list of sets for speed? Or just count coverage?
        # Let's use a numpy array of counts.
        # count[t_idx] = number of blocks in indices that cover t_idx
        
        n_sol = len(indices)
        if n_sol == 0:
            return indices
            
        # Initialize counts
        counts = np.zeros(n_t, dtype=np.int32)
        for idx in indices:
            counts[coverage_matrix[idx]] += 1
            
        changed = True
        while changed:
            changed = False
            new_indices = []
            for idx in indices:
                # Check if this block is the *only* provider for any t-subset
                # If counts[t_idx] == 1 for any t_idx in coverage_matrix[idx], 
                # then this block is NOT redundant.
                # If counts[t_idx] > 1 for ALL t_idx in coverage_matrix[idx], 
                # then this block IS redundant.
                
                t_indices = coverage_matrix[idx]
                # If any count is 1, it's essential.
                if np.any(counts[t_indices] == 1):
                    new_indices.append(idx)
                else:
                    # It is redundant, remove it
                    changed = True
                    # Decrement counts for its subsets
                    counts[t_indices] -= 1
                    
            indices = new_indices
            
        return indices

    blocks_found_indices = reduce_solution(blocks_found_indices)
    
    # Local Search: Try to swap blocks to reduce size further
    # Strategy: 
    # 1. For each block B in solution:
    #    a. Remove B.
    #    b. Find uncovered subsets U.
    #    c. If U is empty, we saved 1 block. (This should have been caught by reduction, but maybe not if order matters? No, reduction handles all.)
    #    d. If U is not empty, find a block B' not in solution that covers the most of U.
    #    e. Add B'.
    #    f. Check if any *other* block B'' in solution (excluding B') is now redundant.
    #    g. If so, remove B''.
    #    h. If total size decreased, accept the change.
    
    # To make this fast, we need to be careful.
    # Given time constraints, we might just do a few passes.
    
    current_indices = blocks_found_indices
    current_set = set(current_indices)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Iterate over a copy of indices to avoid modification during iteration
        for i in range(len(current_indices)):
            if time.time() > deadline:
                break
                
            idx_to_remove = current_indices[i]
            
            # Temporarily remove
            new_indices = [x for x in current_indices if x != idx_to_remove]
            new_set = set(new_indices)
            
            # Check coverage
            # We need to know which t-subsets are covered by new_indices
            # This is expensive to recompute from scratch.
            # Let's maintain counts? 
            # For simplicity and safety, let's just do a quick check if this is worth it.
            # Given the size (v=19), recomputing coverage for ~300 blocks is fast.
            
            # Rebuild coverage counts for new_indices
            counts = np.zeros(n_t, dtype=np.int32)
            for idx in new_indices:
                counts[coverage_matrix[idx]] += 1
                
            # Uncovered subsets
            uncovered = np.where(counts == 0)[0]
            
            if len(uncovered) == 0:
                # We removed a block and coverage is still complete!
                # This shouldn't happen if reduce_solution was perfect, but just in case.
                current_indices = new_indices
                improved = True
                continue
            
            # Find best replacement
            # We want a block that covers the most of 'uncovered'
            # gains = sum of (coverage_matrix[j, k] is in uncovered) for all j not in new_set
            # This is still O(n_k * t_per_block)
            
            # Optimization: Only consider blocks that intersect with uncovered?
            # Actually, any block can be a candidate.
            
            # Vectorized gain calculation for replacement
            # Create a boolean mask of uncovered subsets
            unc_mask = np.zeros(n_t, dtype=bool)
            unc_mask[uncovered] = True
            
            # Calculate how many uncovered subsets each candidate block covers
            # taken = np.take(unc_mask, coverage_matrix) -> (n_k, t_per_block)
            # gains = np.sum(taken, axis=1)
            
            taken = np.take(unc_mask, coverage_matrix)
            gains = np.sum(taken, axis=1)
            
            # Exclude blocks already in new_set
            for idx in new_set:
                gains[idx] = -1
                
            best_idx = np.argmax(gains)
            
            if gains[best_idx] == -1 or gains[best_idx] == 0:
                # No good replacement
                continue
                
            # Add best_idx
            candidate_indices = new_indices + [best_idx]
            candidate_set = new_set | {best_idx}
            
            # Now check if we can remove *another* block from candidate_indices
            # to make it smaller than original (len(current_indices))
            # Original size: len(current_indices)
            # Candidate size so far: len(current_indices) (removed 1, added 1)
            # We need to remove 1 more to improve.
            
            # Check redundancy in candidate_indices
            # Build counts for candidate_indices
            counts_cand = np.zeros(n_t, dtype=np.int32)
            for idx in candidate_indices:
                counts_cand[coverage_matrix[idx]] += 1
                
            redundant_found = False
            final_candidate_indices = []
            
            for idx in candidate_indices:
                t_indices = coverage_matrix[idx]
                # If any t_idx is covered only by this block, it's essential
                if np.any(counts_cand[t_indices] == 1):
                    final_candidate_indices.append(idx)
                else:
                    # Redundant, remove it
                    redundant_found = True
                    counts_cand[t_indices] -= 1
                    
            if redundant_found and len(final_candidate_indices) < len(current_indices):
                current_indices = final_candidate_indices
                current_set = set(current_indices)
                improved = True
                # Restart the loop from the beginning as indices changed
                break
                
    return [list(k_subs[idx]) for idx in current_indices]
