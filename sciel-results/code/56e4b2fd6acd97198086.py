import itertools
import time
import random
import numpy as np
from scipy.sparse import csr_matrix

# Global cache for precomputed data
_PRECOMPUTED = {}

def _precompute(v, k, t):
    key = (v, k, t)
    if key in _PRECOMPUTED:
        return _PRECOMPUTED[key]
        
    points = list(range(v))
    all_t_subsets = list(itertools.combinations(points, t))
    N_t = len(all_t_subsets)
    subset_to_id = {s: i for i, s in enumerate(all_t_subsets)}
    
    all_k_blocks = list(itertools.combinations(points, k))
    N_k = len(all_k_blocks)
    
    # Build sparse matrix M (N_k x N_t)
    # M[i, j] = 1 if block i covers subset j
    rows = []
    cols = []
    
    # Precompute t-subsets for each k-block to speed up row generation?
    # Or just do it in the loop. N_k * C(k,t) is small enough.
    for i, block in enumerate(all_k_blocks):
        for sub in itertools.combinations(block, t):
            j = subset_to_id[sub]
            rows.append(i)
            cols.append(j)
            
    data = np.ones(len(rows), dtype=np.int8)
    M = csr_matrix((data, (rows, cols)), shape=(N_k, N_t))
    
    _PRECOMPUTED[key] = (M, all_k_blocks, subset_to_id, N_t)
    return _PRECOMPUTED[key]

def construct(v, k, t):
    start_time = time.time()
    # 150s limit, leave 20s buffer
    deadline = start_time + 130.0
    
    M, all_k_blocks, subset_to_id, N_t = _precompute(v, k, t)
    N_k = M.shape[0]
    
    best_blocks = None
    best_count = float('inf')
    
    # Multi-start loop
    while time.time() < deadline:
        blocks, valid = _greedy_reduced_once(M, all_k_blocks, N_t, N_k)
        
        if valid:
            if len(blocks) < best_count:
                best_count = len(blocks)
                best_blocks = blocks
                # Early stop if we hit a very good bound?
                # Best known for 19,8,5 is 335.
                if best_count <= 335:
                    # Could try to improve further, but 335 is good.
                    # Let's keep searching for a bit if time allows.
                    pass
        
        # Check time
        if time.time() > deadline - 5.0:
            break
            
    if best_blocks is None:
        # Fallback
        blocks, _ = _greedy_reduced_once(M, all_k_blocks, N_t, N_k)
        return blocks
        
    # Convert internal block indices back to point lists if necessary?
    # all_k_blocks contains tuples of points.
    # The return type should be list of lists of ints.
    return [list(b) for b in best_blocks]

def _greedy_reduced_once(M, all_k_blocks, N_t, N_k):
    """
    Runs one instance of Greedy Construction followed by Reduction.
    M: Sparse incidence matrix (N_k x N_t)
    all_k_blocks: List of tuples representing the blocks
    """
    counts = np.zeros(N_t, dtype=np.int32)
    block_indices = []
    
    # Greedy Phase
    while True:
        # Check if all covered
        if np.all(counts > 0):
            break
            
        # Identify uncovered subsets
        uncovered_mask = (counts == 0).astype(np.int8)
        
        # Calculate gain for each block: number of uncovered subsets it covers
        # M is N_k x N_t. uncovered_mask is 1 x N_t.
        # gains is N_k x 1.
        gains = M.dot(uncovered_mask)
        
        max_gain = np.max(gains)
        if max_gain == 0:
            # Should not happen if not all covered, unless M is broken
            # But if there are uncovered subsets, there must be a block covering them
            # since v >= k >= t
            break
            
        # Find candidates with max gain
        # To avoid large lists, we can just pick one randomly from the top
        # or use a secondary criterion.
        candidates = np.where(gains == max_gain)[0]
        
        # Secondary criterion: Maximize coverage of subsets with count == 1
        # This makes the solution more robust for reduction.
        if len(candidates) > 1:
            # We pick a few random candidates and evaluate secondary score
            # Or just pick one randomly to maintain diversity in multi-start
            # Let's pick randomly for diversity, but maybe bias towards 
            # blocks that cover more 'critical' (count=1) subsets.
            
            # For efficiency, let's just pick one randomly.
            # The multi-start aspect handles the optimization.
            selected_idx = random.choice(candidates)
        else:
            selected_idx = candidates[0]
            
        block_indices.append(selected_idx)
        
        # Update counts
        # M[selected_idx] is a sparse row
        counts += M[selected_idx].toarray().flatten()
        
    # Reduction Phase
    # We have block_indices. We need to check if any can be removed.
    # A block is removable if for all subsets s in block, counts[s] > 1.
    
    # To speed up reduction, we can use the sparse matrix.
    # But counts change.
    
    # Convert block_indices to a set for fast lookup? No, we just iterate.
    # We need to map block_idx -> list of subset_ids it covers.
    # We can extract this from M.
    
    # Let's create a list of lists of subset IDs for the selected blocks.
    # M is CSR. M.indptr, M.indices.
    
    # For each selected block index, get its non-zero indices (subset IDs)
    block_subsets = []
    for b_idx in block_indices:
        start = M.indptr[b_idx]
        end = M.indptr[b_idx + 1]
        subsets = M.indices[start:end]
        block_subsets.append(subsets)
        
    # Now reduce
    changed = True
    # We need to maintain counts correctly.
    # Current counts are based on all blocks.
    
    # To check if a block is redundant:
    # min(counts[subsets]) > 1
    
    # Iterate through blocks in reverse
    i = len(block_indices) - 1
    while i >= 0:
        subsets = block_subsets[i]
        # Check if any count is 1
        # If min(counts[subsets]) == 1, it's essential.
        # If min > 1, it's redundant.
        
        # counts is a numpy array. counts[subsets] gives values.
        if np.min(counts[subsets]) > 1:
            # Remove block i
            # Decrement counts for its subsets
            counts[subsets] -= 1
            block_indices.pop(i)
            block_subsets.pop(i)
            # Do not decrement i, check same index again
        else:
            i -= 1
            
    # Re-check reduction? Sometimes removing a block makes another one redundant.
    # The loop above handles this because it doesn't decrement i on removal.
    # But it iterates from end to start.
    # If we remove block i, block i-1 is now at index i-1.
    # The loop continues.
    # However, removing a block *increases* the chance of other blocks being redundant?
    # No, removing a block *decreases* counts.
    # If a block was essential (min count 1), and we remove *another* block,
    # the counts drop further. It might become redundant? No, counts drop,
    # so min count might drop to 0? No, we only remove if min > 1.
    # So counts only decrease when we remove a block.
    # If a block was redundant (min > 1), and we remove another block,
    # its min count might drop to 1, making it essential.
    # So a single pass from end to start is NOT sufficient for full reduction.
    # We need to repeat until no changes.
    
    # Let's wrap the reduction in a while loop.
    
    # Reset for proper iterative reduction
    # We already did one pass. Let's do a proper iterative one.
    
    # Re-initialize counts and lists? No, we can just re-run the check.
    # But we need to ensure counts are consistent.
    
    # Let's restart the reduction process cleanly on the current state.
    # Actually, the previous loop modified counts and lists.
    # Let's just run a full reduction loop.
    
    # To be safe and simple:
    # 1. Rebuild block_subsets from current block_indices
    # 2. Rebuild counts from scratch? No, that's expensive.
    # 3. Just iterate while changed.
    
    # The previous loop:
    # i = len - 1
    # while i >= 0:
    #   if redundant: remove, don't dec i
    #   else: dec i
    # This effectively scans the list. If it removes one, it re-checks the same position (which is now the next element from the right).
    # Does it miss things?
    # Example: Blocks A, B, C.
    # C redundant. Remove C. Counts drop.
    # Now check B (which was at i-1, now at i-1? No, C was at i. B is at i-1).
    # After removing C, list is [A, B]. i is still index of C (which was last).
    # Wait, if we pop(i), the element at i is now the one that was at i+1?
    # No, pop(i) removes element at i. The list shrinks.
    # If we are at i (last index), and we pop it, the new last index is i-1.
    # We should decrement i?
    # If we pop i, the next element to check should be i-1 (the new last).
    # So if we pop, we should NOT decrement i, because the list shifted?
    # No.
    # List: [A, B, C]. i=2 (C).
    # Pop C. List: [A, B].
    # Next check should be B (index 1).
    # If we don't decrement i, i is still 2. But list length is 2. Index 2 is out of bounds.
    # So we MUST decrement i if we pop?
    # If we pop i, the element that was at i+1 (none) is gone.
    # The element that was at i-1 is now at i-1.
    # So we should decrement i to check i-1.
    
    # Let's re-verify.
    # Standard pattern:
    # i = len - 1
    # while i >= 0:
    #   if condition:
    #     pop(i)
    #     # do not decrement i?
    #     # If we pop i, the next element to examine is the one that is now at i (which was i+1)? No, i+1 doesn't exist or is gone.
    #     # Actually, if we iterate from end to start, and we remove an item, the items to the left are unaffected in their indices.
    #     # So we should just continue to i-1?
    #     # If we remove item at i, the item at i-1 is still at i-1.
    #     # So we should decrement i.
    #   else:
    #     i -= 1
    
    # Let's test: [A, B, C]. i=2.
    # Remove C. List [A, B].
    # Next we want to check B (index 1).
    # If we decrement i, i becomes 1. Correct.
    
    # So the previous code `i -= 1` in the else branch, and NO decrement in the if branch was WRONG.
    # It should be:
    # if redundant:
    #    pop(i)
    #    i -= 1 # Move to next (left) element
    # else:
    #    i -= 1
    
    # Wait, if we pop i, and then i -= 1, we skip the element that shifted into position i?
    # No, nothing shifts into position i from the right.
    # The list shrinks.
    # So yes, i -= 1 is correct in both cases.
    
    # However, we also need to handle the case where removing a block makes *another* block redundant that we already passed.
    # Since we go right-to-left, we haven't passed the left ones yet.
    # But what if removing C makes A redundant?
    # We check C, remove it.
    # Check B.
    # Check A.
    # If A is now redundant, we remove it.
    # This works for a single pass.
    # But what if removing A (last step) makes B redundant?
    # We already checked B.
    # So one pass is not enough.
    
    # Let's implement the full while-changed loop properly.
    
    # Re-do reduction cleanly
    # We have block_indices and counts.
    # But counts might be inconsistent if we messed up the previous logic.
    # Let's just recalculate counts from block_indices to be safe?
    # N_t is 11k. Block count ~300.
    # Recalculating counts: 300 * 56 ops = 16k ops. Fast.
    
    # Let's restart the reduction phase from scratch to be safe.
    counts = np.zeros(N_t, dtype=np.int32)
    for b_idx in block_indices:
        counts += M[b_idx].toarray().flatten()
        
    # Rebuild block_subsets
    block_subsets = []
    for b_idx in block_indices:
        start = M.indptr[b_idx]
        end = M.indptr[b_idx + 1]
        block_subsets.append(M.indices[start:end])
        
    changed = True
    while changed:
        changed = False
        i = len(block_indices) - 1
        while i >= 0:
            subsets = block_subsets[i]
            if np.min(counts[subsets]) > 1:
                # Redundant
                counts[subsets] -= 1
                block_indices.pop(i)
                block_subsets.pop(i)
                changed = True
                i -= 1 # Move to next
            else:
                i -= 1
                
    # Final validation
    if np.any(counts == 0):
        return [], False
        
    # Convert indices to actual blocks
    final_blocks = [all_k_blocks[idx] for idx in block_indices]
    return final_blocks, True

