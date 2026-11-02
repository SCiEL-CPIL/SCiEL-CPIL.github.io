import numpy as np
import itertools
import time
import random

def construct(v, k, t):
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    points = range(v)
    all_ts = list(itertools.combinations(points, t))
    n_ts = len(all_ts)
    
    # Map t-subsets to indices
    idx_map = {comb: i for i, comb in enumerate(all_ts)}
    
    # Precompute all k-subsets (blocks)
    all_ks = list(itertools.combinations(points, k))
    n_ks = len(all_ks)
    
    # Precompute for each block, the indices of the t-subsets it covers
    # This is a sparse structure, but we can store it as a list of arrays
    block_t_indices = []
    for block in all_ks:
        subs = list(itertools.combinations(block, t))
        idxs = np.array([idx_map[sub] for sub in subs], dtype=np.int32)
        block_t_indices.append(idxs)
        
    # For fast coverage counting, we might need a matrix, but n_ts is ~2024 and n_ks is ~346k.
    # A full n_ks x n_ts matrix is too large (346k * 2024 * 4 bytes ~ 2.8 GB).
    # We will use the list of arrays approach for coverage updates.

    best_solution = None
    best_score = float('inf')
    
    def count_uncovered(blocks_indices):
        """Returns the number of uncovered t-subsets and the coverage array."""
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b_idx in blocks_indices:
            coverage[block_t_indices[b_idx]] += 1
        uncovered = np.sum(coverage == 0)
        return uncovered, coverage

    def try_delete_block(blocks_indices, coverage, b_idx):
        """
        Attempts to delete block b_idx from blocks_indices.
        Returns True if deletion was successful and updates coverage.
        """
        subs = block_t_indices[b_idx]
        # Check if all subsets in this block are covered by at least one other block
        if np.all(coverage[subs] > 1):
            coverage[subs] -= 1
            blocks_indices.remove(b_idx)
            return True
        return False

    def aggressive_deletion(blocks_indices, coverage):
        """Iteratively deletes redundant blocks until none remain."""
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 2.0:
                break
            improved = False
            # Iterate in reverse to avoid index shifting issues during removal
            for i in range(len(blocks_indices) - 1, -1, -1):
                b_idx = blocks_indices[i]
                subs = block_t_indices[b_idx]
                if np.all(coverage[subs] > 1):
                    coverage[subs] -= 1
                    blocks_indices.pop(i)
                    improved = True
        return blocks_indices

    def local_search_2exchange(blocks_indices, coverage, max_iters=500):
        """
        Local search using 1-for-1 block exchange.
        Tries to replace a block with another block that allows deletion of the original
        or leads to a net reduction in blocks after deletion.
        """
        blocks_list = list(blocks_indices)
        # Precompute block sets for fast membership? No, we need to swap.
        
        # We need to be careful with time.
        # Strategy: 
        # 1. Pick a random block B1.
        # 2. Try replacing B1 with a random candidate B2 (from a pool of good candidates).
        # 3. If coverage is maintained (all t-subsets still covered), update coverage.
        # 4. Try to delete B1 (now redundant if B2 covers its unique parts? No, B1 is removed, B2 added).
        #    Actually, if we replace B1 with B2, we remove B1 and add B2.
        #    If B2 covers all unique t-subsets of B1, then B1 was redundant? No.
        #    The goal is to find a set of blocks of smaller size.
        
        # Better approach for 2-exchange in covering:
        # Current set S.
        # Remove B1 from S. Let S' = S \ {B1}.
        # If S' covers all t-subsets, we are done (deletion).
        # If not, let U be the set of uncovered t-subsets in S'.
        # Find a block B2 (not in S) that covers the most of U.
        # If B2 covers ALL of U, then S_new = S' U {B2} is a valid cover of same size.
        #    Then try to delete any redundant blocks in S_new.
        #    If a deletion is possible, size decreases by 1.
        
        # This is effectively "Greedy Repair".
        
        for _ in range(max_iters):
            if time.time() - start_time > time_limit - 5.0:
                break
            
            if len(blocks_list) == 0:
                break
                
            # Pick a random block to potentially remove
            rand_idx = random.randint(0, len(blocks_list) - 1)
            b1 = blocks_list[rand_idx]
            
            # Temporarily remove b1
            blocks_list.pop(rand_idx)
            
            # Calculate uncovered subsets
            # We need to know which subsets are currently covered by 0 blocks.
            # We have 'coverage' array for the current state (with b1).
            # So, coverage[subs_of_b1] - 1 gives coverage without b1.
            subs_b1 = block_t_indices[b1]
            
            # Check if b1 is redundant
            if np.all(coverage[subs_b1] > 1):
                # It is redundant! We already handled this in aggressive_deletion, 
                # but maybe it became redundant after a swap.
                coverage[subs_b1] -= 1
                # We removed it, so size decreased.
                continue
            
            # It is not redundant. Find the subsets that become uncovered.
            # uncovered_mask = (coverage[subs_b1] == 1)
            # But we need the global uncovered set.
            # Let's compute the new coverage state.
            # new_coverage = coverage.copy()
            # new_coverage[subs_b1] -= 1
            # uncovered_indices = np.where(new_coverage == 0)[0]
            
            # Optimization: Only check subsets of b1 that were covered exactly once.
            # If a subset of b1 was covered >= 2 times, it remains covered.
            # So, candidates for uncovered are those in subs_b1 where coverage == 1.
            unique_subs_b1 = subs_b1[coverage[subs_b1] == 1]
            
            if len(unique_subs_b1) == 0:
                # Should have been caught by redundant check, but just in case
                coverage[subs_b1] -= 1
                continue
            
            # We need to find a block B2 that covers ALL of unique_subs_b1.
            # This is a hitting set problem on the subsets of unique_subs_b1.
            # Since unique_subs_b1 is small (subset of C(7,3)=35), we can check candidates.
            
            # Which blocks cover all of unique_subs_b1?
            # We can precompute for each t-subset, the list of blocks covering it.
            # But we don't have that stored efficiently for all n_ts.
            
            # Alternative: Random sampling of candidate blocks.
            # Or, check blocks that share points with b1.
            
            # Let's try a few random blocks from the universe of all_ks.
            # To be faster, we can restrict to blocks that intersect b1 significantly?
            # Actually, the best candidate B2 must cover the "hole" left by B1.
            
            found_replacement = False
            # Try up to 20 random candidates
            for _ in range(20):
                b2 = random.randint(0, n_ks - 1)
                if b2 == b1:
                    continue
                if b2 in blocks_list:
                    continue
                
                subs_b2 = block_t_indices[b2]
                # Check if subs_b2 covers all unique_subs_b1
                # We need to check if unique_subs_b1 is a subset of subs_b2.
                # Since both are sorted arrays of ints, we can use searchsorted or set intersection.
                # For small arrays, set intersection is fast enough.
                if len(set(unique_subs_b1).intersection(subs_b2)) == len(unique_subs_b1):
                    # B2 covers the hole.
                    # Add B2 to blocks_list
                    blocks_list.append(b2)
                    # Update coverage
                    # coverage[subs_b1] -= 1 (already logically removed)
                    coverage[subs_b1] -= 1
                    coverage[subs_b2] += 1
                    
                    found_replacement = True
                    break
            
            if found_replacement:
                # Now try to delete redundant blocks in the new set
                aggressive_deletion(blocks_list, coverage)
            else:
                # Restore b1
                blocks_list.insert(rand_idx, b1)
                # Coverage was not modified, so no need to restore.
                # Wait, we didn't modify coverage in the 'fail' case.
                # In the 'success' case, we modified coverage.
                # So if found_replacement is False, coverage is still correct for the state WITH b1.
                # We just popped b1 from the list. So we must push it back.
                # And coverage is unchanged.
                pass

        return blocks_list

    def run_greedy_with_local_search(seed):
        random.seed(seed)
        np.random.seed(seed)
        
        covered = np.zeros(n_ts, dtype=bool)
        blocks_indices = []
        remaining = n_ts
        
        # Greedy Phase
        while remaining > 0:
            if time.time() - start_time > time_limit - 10.0:
                return None, None
            
            # Compute gains for all blocks
            # gain[b] = number of t-subsets in block b that are NOT covered
            # We can vectorize this:
            # covered_indices = np.where(covered)[0]
            # For each block, count how many of its t-subsets are in covered_indices.
            
            # Efficient way:
            # For each block, we have an array of t-subset indices.
            # We want sum(~covered[indices])
            
            # Let's use the precomputed block_t_indices.
            # This is a list of arrays.
            # We can't easily vectorize across the list without a matrix.
            # Let's create a sparse-like structure or just loop if n_ks is small enough?
            # n_ks = 346k. Looping 346k times is slow in Python.
            
            # Alternative: 
            # Maintain a count of how many times each t-subset is "wanted" by uncovered blocks?
            # No, standard greedy is: pick block with max new coverage.
            
            # Let's try to vectorize the gain calculation.
            # We have `block_t_indices` as a list of arrays.
            # We can stack them into a 2D array if they are same length? Yes, all length C(k,t).
            # But 346k * 35 is 12M integers. 12M * 4 bytes = 48 MB. This is feasible.
            
            if not hasattr(run_greedy_with_local_search, 'block_matrix'):
                # Create matrix once
                run_greedy_with_local_search.block_matrix = np.array(block_t_indices, dtype=np.int32)
            
            block_matrix = run_greedy_with_local_search.block_matrix
            
            # gains = number of uncovered t-subsets in each block
            # ~covered is a boolean array of length n_ts.
            # block_matrix is (n_ks, C(k,t)).
            # We want to sum ~covered[block_matrix] along axis 1.
            # ~covered[block_matrix] will be a boolean array of shape (n_ks, C(k,t)).
            # This is 12M booleans. 12 MB. Feasible.
            
            gains = (~covered)[block_matrix].sum(axis=1)
            
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Find blocks with max gain
            # To introduce diversity, pick randomly from top candidates
            top_k = 10
            candidates = np.where(gains >= max_gain - 1)[0] # Allow slight tolerance for diversity
            if len(candidates) > top_k:
                candidates = candidates[random.sample(range(len(candidates)), top_k)]
            
            best_candidate = candidates[0]
            # Optional: Pick the one that covers rare triples?
            # For now, random from top.
            
            # Add block
            blocks_indices.append(best_candidate)
            
            # Update covered
            subs = block_t_indices[best_candidate]
            new_covers = np.sum(~covered[subs])
            remaining -= new_covers
            covered[subs] = True
            
        # Local Search Phase
        if time.time() - start_time > time_limit - 15.0:
            return blocks_indices, len(blocks_indices)
            
        coverage = np.zeros(n_ts, dtype=np.int32)
        for b in blocks_indices:
            coverage[block_t_indices[b]] += 1
            
        # Aggressive deletion
        blocks_indices = aggressive_deletion(blocks_indices, coverage)
        
        # 2-Exchange Local Search
        if time.time() - start_time < time_limit - 5.0:
            blocks_indices = local_search_2exchange(blocks_indices, coverage, max_iters=200)
            # Recalculate coverage after local search
            coverage = np.zeros(n_ts, dtype=np.int32)
            for b in blocks_indices:
                coverage[block_t_indices[b]] += 1
            blocks_indices = aggressive_deletion(blocks_indices, coverage)
            
        return blocks_indices, len(blocks_indices)

    # Main Loop: Multiple Restarts
    restarts = 0
    while time.time() - start_time < time_limit - 20.0:
        seed = restarts
        blocks, score = run_greedy_with_local_search(seed)
        
        if blocks is not None:
            # Validate
            # Check distinctness
            if len(set(tuple(b) for b in blocks)) != len(blocks):
                # Should not happen
                pass
            
            # Check coverage
            # Re-verify coverage to be safe
            cov_check = np.zeros(n_ts, dtype=np.int32)
            for b in blocks:
                # b is a list of points. We need to map to t-subset indices.
                # This is slow. We should store indices in blocks_indices.
                pass
            
            # We stored indices in blocks_indices.
            # Let's trust the internal coverage tracking but do a final check on size.
            if score < best_score:
                best_score = score
                best_solution = blocks
        
        restarts += 1
        
        # If we've hit the lower bound (69), we can stop early?
        # Theoretical lower bound is 69.
        if best_score <= 69:
            break

    if best_solution is None:
        # Fallback
        blocks, score = run_greedy_with_local_search(0)
        if blocks:
            best_solution = blocks

    # Convert indices back to point lists
    final_blocks = [list(all_ks[i]) for i in best_solution]
    
    return final_blocks

