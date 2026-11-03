import itertools
import time
import random
import numpy as np
import sys

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s margin for safety

    # 1. Precompute all t-subsets and map to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subsets)}
    n_t = len(t_subsets)

    # 2. Precompute all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    n_k = len(k_subsets)

    # 3. Precompute block_t_indices: (n_k, C(k, t))
    # For each block (k-subset), which t-subsets does it cover?
    # We need the indices of these t-subsets in the global t_sub_to_idx map.
    
    # Helper to get t-subset indices for a given k-subset
    # This is done once.
    # Optimization: We can precompute the local t-subsets of a k-set relative to 0..k-1
    # and then map them to the global set.
    
    # However, mapping is the bottleneck. 
    # Let's use a faster method:
    # For a block B = (b0, ..., bk-1), the t-subsets are combinations of indices.
    # We can precompute the list of local t-subsets (indices into 0..k-1).
    
    local_t_subs = list(itertools.combinations(range(k), t))
    n_t_in_k = len(local_t_subs)
    
    block_t_indices = np.empty((n_k, n_t_in_k), dtype=np.int32)
    
    # Fill block_t_indices
    # We can parallelize this or just loop, but Python loops are slow.
    # Since n_k is ~75k and n_t_in_k is 56, total ~4M operations.
    # We can try to vectorize the lookup.
    
    # Alternative: Use a dictionary lookup but batch it?
    # Or just use a fast loop. 4M iterations in Python is ~0.5-1s. Acceptable.
    
    for i, block in enumerate(k_subsets):
        for j, lc in enumerate(local_t_subs):
            # Create the tuple of points
            t_sub = tuple(block[p] for p in lc)
            block_t_indices[i, j] = t_sub_to_idx[t_sub]
            
    # Precompute t_to_blocks: list of lists of block indices
    # This is the inverse mapping.
    # n_t is C(19,5) = 11628.
    # Total entries is n_k * n_t_in_k = 4.2M.
    # We can build this by iterating over the flat array.
    
    t_to_blocks = [[] for _ in range(n_t)]
    # Vectorized approach to populate lists is hard in pure Python/numpy without object arrays.
    # Let's stick to a loop but it's only 4M appends.
    # Actually, we can use a temporary array of pairs (t_idx, block_idx) and sort/group.
    # Or just loop. 4M appends is fast enough (<1s).
    
    for i in range(n_k):
        for t_idx in block_t_indices[i]:
            t_to_blocks[t_idx].append(i)
            
    # Convert t_to_blocks to numpy arrays for faster intersection if needed?
    # Or keep as lists for set operations.
    # For local search, we often need intersection of sets.
    # Let's keep as lists for now, or convert to sets if used frequently.
    # In local search, we check candidates for unique t-subsets.
    
    # 4. Greedy Construction with NumPy acceleration
    
    def time_left():
        return deadline - time.time()

    def greedy_construct():
        if time_left() < 1:
            return None
        
        # block_scores[i] = number of currently uncovered t-subsets covered by block i
        # Initially, all t-subsets are uncovered.
        # Score of block i is just n_t_in_k (C(k,t)) because it covers that many distinct t-subsets.
        # Wait, if we select blocks, scores decrease.
        
        block_scores = np.full(n_k, n_t_in_k, dtype=np.int32)
        uncovered = np.ones(n_t, dtype=bool) # True if uncovered
        
        selected_blocks = []
        selected_set = set()
        
        while np.any(uncovered) and time_left() > 5:
            # Find block with max score
            # We should ignore already selected blocks.
            # Mask out selected blocks
            # Creating a mask of n_k every time is O(n_k).
            # 75k is small.
            
            # Optimization: Instead of masking, we can set score to -1 for selected.
            # But we need to restore if we backtrack? No, greedy is forward only.
            
            # Let's just find max.
            # To avoid selecting same block, set score to -1 after selection.
            
            best_idx = np.argmax(block_scores)
            best_score = block_scores[best_idx]
            
            if best_score <= 0:
                break # No improvement possible? Or all covered?
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update scores
            # For each t-subset covered by best_idx that is currently uncovered:
            # 1. Mark it as covered (uncovered[t_idx] = False)
            # 2. Decrement score of all other blocks covering t_idx
            
            t_idxs = block_t_indices[best_idx]
            # Only process those that are still uncovered
            active_t_idxs = t_idxs[uncovered[t_idxs]]
            
            if len(active_t_idxs) == 0:
                block_scores[best_idx] = -1
                continue
                
            # Mark as covered
            uncovered[active_t_idxs] = False
            
            # Decrement scores of other blocks
            # For each t_idx in active_t_idxs, we need to decrement block_scores for blocks in t_to_blocks[t_idx]
            # excluding best_idx and selected ones.
            
            # This is the expensive part.
            # Vectorized update:
            # We can't easily do sparse matrix subtraction here without scipy.sparse.
            # Let's use a loop over active_t_idxs.
            # active_t_idxs is at most n_t_in_k (56).
            # For each, t_to_blocks[t_idx] has ~56 entries (symmetry).
            # So ~56 * 56 = 3136 operations per block selection.
            # 335 blocks * 3136 = ~1M operations. Very fast.
            
            for t_idx in active_t_idxs:
                # Get blocks covering this t_idx
                blocks_covering = t_to_blocks[t_idx]
                # We need to decrement block_scores for these blocks
                # But only if they are not selected and not the current best_idx
                # And only if they cover this t_idx (which they do by definition)
                
                # We can just decrement the numpy array indices
                # But we must skip best_idx and selected blocks.
                # Since we set best_idx score to -1 later, and selected blocks have -1,
                # we can just decrement and then check?
                # No, decrementing a -1 makes it -2, which is fine, it stays "bad".
                # But we must not decrement the score of best_idx itself?
                # Actually, best_idx is in t_to_blocks[t_idx].
                # If we decrement it, it goes from n_t_in_k to n_t_in_k - 1.
                # But we are about to set it to -1 anyway.
                
                # So, just decrement all blocks in t_to_blocks[t_idx]
                # But be careful: if a block is already selected, its score is -1.
                # Decrementing it makes it -2. That's fine.
                
                # Vectorized decrement
                block_scores[blocks_covering] -= 1
                
            # Mark best_idx as selected (score = -1)
            block_scores[best_idx] = -1
            
        return selected_blocks

    def local_search_v2(initial_blocks):
        if time_left() < 5:
            return initial_blocks
            
        current_blocks = list(initial_blocks)
        current_set = set(current_blocks)
        
        # Compute cover counts
        # cover_count[t_idx] = number of blocks in current_set covering t_idx
        cover_count = np.zeros(n_t, dtype=np.int32)
        
        for b in current_blocks:
            cover_count[block_t_indices[b]] += 1
            
        def try_reduce():
            if time_left() < 2:
                return False
            changed = False
            # Find blocks where all covered t-subsets have count > 1
            # i.e., block is redundant.
            
            # We can vectorize this check?
            # For each block, check if np.all(cover_count[block_t_indices[b]] > 1)
            # This is n_k * n_t_in_k checks.
            # 75k * 56 = 4M checks.
            
            # Let's loop through blocks.
            new_blocks = []
            for b in current_blocks:
                if time_left() < 2:
                    # Abort reduction if time is tight, keep current state
                    break
                ts = block_t_indices[b]
                if np.all(cover_count[ts] > 1):
                    # Redundant
                    cover_count[ts] -= 1
                    current_set.remove(b)
                    changed = True
                else:
                    new_blocks.append(b)
                    
            if changed:
                current_blocks[:] = new_blocks
                return True
            return False
            
        # Initial reduction
        try_reduce()
        
        # Local Search Loop
        max_iterations = 200
        no_improve_count = 0
        
        for _ in range(max_iterations):
            if time_left() < 10:
                break
                
            if not try_reduce():
                # Try a swap
                if not current_blocks:
                    break
                    
                # Pick a random block to remove
                b_out = random.choice(current_blocks)
                
                # Find t-subsets uniquely covered by b_out
                ts_out = block_t_indices[b_out]
                unique_ts = ts_out[cover_count[ts_out] == 1]
                
                if len(unique_ts) == 0:
                    # Should have been reduced
                    continue
                    
                # Find candidate blocks that cover all unique_ts
                # Candidates must cover unique_ts[0]
                candidates = set(t_to_blocks[unique_ts[0]])
                candidates -= current_set
                
                for t_idx in unique_ts[1:]:
                    candidates &= set(t_to_blocks[t_idx])
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Pick a random candidate
                b_in = random.choice(list(candidates))
                
                # Try the swap
                # Save state
                old_blocks = list(current_blocks)
                old_set = set(current_blocks)
                old_cover = cover_count.copy()
                
                # Apply swap
                cover_count[ts_out] -= 1
                current_set.remove(b_out)
                current_blocks.remove(b_out)
                
                ts_in = block_t_indices[b_in]
                cover_count[ts_in] += 1
                current_set.add(b_in)
                current_blocks.append(b_in)
                
                # Reduce
                try_reduce()
                
                if len(current_blocks) < len(old_blocks):
                    no_improve_count = 0
                else:
                    # Revert
                    current_blocks[:] = old_blocks
                    current_set = old_set
                    cover_count[:] = old_cover
                    no_improve_count += 1
                    if no_improve_count > 50:
                        break
            else:
                no_improve_count = 0
                
        try_reduce()
        return current_blocks

    best_solution = None
    best_len = float('inf')
    rng = random.Random(42)
    
    # Run multiple greedy restarts
    # The greedy step is fast now.
    # Local search is the bottleneck.
    
    while time_left() > 20:
        sol = greedy_construct()
        if sol is None:
            break
            
        if len(sol) < best_len:
            best_len = len(sol)
            best_solution = sol
            
            # Local Search
            if time_left() > 30:
                improved_sol = local_search_v2(sol)
                if len(improved_sol) < best_len:
                    best_len = len(improved_sol)
                    best_solution = improved_sol
                    
    if best_solution is None:
        # Fallback
        sol = greedy_construct()
        if sol:
            best_solution = sol
        else:
            return []

    return [k_subs[i] for i in best_solution]

if __name__ == "__main__":
    # Test with small parameters to ensure it runs
    # construct(19, 8, 5)
    pass
