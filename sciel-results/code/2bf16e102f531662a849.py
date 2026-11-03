import itertools
import numpy as np
import scipy.sparse as sp
import time
import random

def construct(v, k, t):
    start_time = time.time()
    # Safety margin: 145s out of 150s
    deadline = start_time + 145.0
    
    # 1. Pre-computation
    # All t-subsets (targets)
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    n_t = len(t_subs)
    
    # All k-subsets (candidate blocks)
    k_subs = list(itertools.combinations(range(v), k))
    n_k = len(k_subs)
    
    # Build sparse matrix A (n_k x n_t)
    # A[i, j] = 1 if t_subs[j] is a subset of k_subs[i]
    row_indices = []
    col_indices = []
    
    # Optimization: Iterate through k-subsets and their t-subsets
    for i, block in enumerate(k_subs):
        block_set = set(block)
        # Generate all t-subsets of this block
        for t_sub in itertools.combinations(block, t):
            j = t_sub_to_idx[t_sub]
            row_indices.append(i)
            col_indices.append(j)
            
    A = sp.csr_matrix((np.ones(len(row_indices), dtype=np.int8), (row_indices, col_indices)), 
                      shape=(n_k, n_t), dtype=np.int8)
    
    # Pre-compute column indices for each block for fast updates
    # A is CSR, so A.indices[start:end] gives the columns for row i
    block_cols = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in range(n_k)]
    
    def greedy_cover(seed_blocks=None, random_tie_break=False):
        """Greedy set cover with optional seed blocks and random tie-breaking."""
        u = np.ones(n_t, dtype=np.int8)
        selected_blocks = []
        selected_set = set()
        
        if seed_blocks:
            for idx in seed_blocks:
                if idx in selected_set: continue
                cols = block_cols[idx]
                u[cols] = 0
                selected_blocks.append(idx)
                selected_set.add(idx)
        
        for _ in range(n_k + 10): # Max possible blocks
            if time.time() > deadline:
                break
            if u.sum() == 0:
                break
                
            # Calculate gain: number of uncovered t-subsets covered by each block
            gains = A @ u
            
            if gains.max() == 0:
                break
                
            # Find indices with max gain
            max_gain = gains.max()
            candidates = np.where(gains == max_gain)[0]
            
            if random_tie_break and len(candidates) > 1:
                best_idx = candidates[random.randint(0, len(candidates)-1)]
            else:
                best_idx = candidates[0]
                
            selected_blocks.append(best_idx)
            selected_set.add(best_idx)
            
            # Update uncovered vector
            cols = block_cols[best_idx]
            u[cols] = 0
            
        return selected_blocks

    def remove_redundant(selected_blocks):
        """Remove blocks that do not uniquely cover any t-subset."""
        if not selected_blocks:
            return selected_blocks
            
        # Count how many times each t-subset is covered
        cover_count = np.zeros(n_t, dtype=np.int32)
        for idx in selected_blocks:
            cover_count[block_cols[idx]] += 1
            
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = False
            to_remove = []
            for i in range(len(selected_blocks)):
                idx = selected_blocks[i]
                cols = block_cols[idx]
                # If any t-subset covered by this block has count 1, it's essential
                # If all have count > 1, it's redundant
                if np.all(cover_count[cols] > 1):
                    to_remove.append(i)
                    
            if to_remove:
                changed = True
                # Remove from end to preserve indices
                for i in sorted(to_remove, reverse=True):
                    idx = selected_blocks[i]
                    cover_count[block_cols[idx]] -= 1
                    selected_blocks.pop(i)
                    
        return selected_blocks

    def local_improve(selected_blocks, max_rounds=20):
        """Try to improve by swapping blocks."""
        best = list(selected_blocks)
        best_size = len(best)
        
        for r in range(max_rounds):
            if time.time() > deadline:
                break
            if len(best) == 0:
                break
            
            improved = False
            # Try to remove each block and replace it with a better one
            for i in range(len(best)):
                if time.time() > deadline:
                    break
                
                current_idx = best[i]
                current_cols = block_cols[current_idx]
                
                # Check if current block is essential
                # We need to know if removing it leaves any t-subset uncovered
                # Let's compute cover counts for the rest
                # This is expensive O(N^2). 
                # Alternative: Just try to find a block that covers the "essential" parts of current_idx
                # A simpler heuristic: 
                # 1. Remove block i
                # 2. Find uncovered subsets
                # 3. Find best block to cover them
                # 4. If new size <= old size, accept
                
                temp_best = [b for j, b in enumerate(best) if j != i]
                
                # Calculate uncovered if we remove block i
                # We need cover counts for temp_best
                # To make this fast, we can maintain a global cover count and just subtract/add
                # But here we are doing a trial swap.
                
                # Let's use a faster check:
                # Is there a block b' (b' != b_i) such that 
                # Cover(All \ {b_i} U {b'}) is valid?
                # This is hard to check without full recount.
                
                # Heuristic: 
                # Find t-subsets covered ONLY by b_i in the current set.
                # If none, b_i is redundant (handled by remove_redundant).
                # If some, find a block b' that covers the most of those "only-covered" subsets.
                
                # To find "only-covered", we need cover counts.
                # Let's do a full recount for the candidate swap only if potential gain is high?
                # No, let's just try random swaps and validate.
                
                # Pick a random candidate block
                candidate_idx = random.randint(0, n_k - 1)
                if candidate_idx == current_idx:
                    continue
                
                # Try swap
                new_block_list = temp_best + [candidate_idx]
                
                # Validate coverage
                # Check if every t-subset is covered
                # We can do this by checking if the union of cols covers all n_t
                # But checking "is every t-subset covered" is equivalent to 
                # "does the sum of indicator vectors >= 1 for all t"?
                
                # Fast check using numpy:
                # Sum the columns of the new blocks? No, that's expensive.
                
                # Let's stick to a simpler local improvement:
                # Just run remove_redundant again after a random perturbation?
                # Or, try to replace a block with one that has higher "density" of essential coverage.
                
                # Given the complexity, let's rely on multiple restarts of Greedy + Remove Redundant.
                # This is often very effective for covering designs.
                
                pass # Skipping complex local search in this iteration to ensure robustness and speed
                
        return best

    # Main Loop: Multiple restarts
    best_cover = None
    best_size = float('inf')
    
    # First run: Deterministic Greedy
    cover = greedy_cover(random_tie_break=False)
    cover = remove_redundant(cover)
    if len(cover) < best_size:
        best_size = len(cover)
        best_cover = cover
    
    # Subsequent runs: Randomized Greedy
    # How many restarts can we do?
    # Greedy takes ~0.1-0.5s. Remove redundant takes ~0.1s.
    # We have ~140s. So ~100-200 restarts.
    
    restarts = 0
    while time.time() < deadline - 2.0: # Leave 2s for final conversion
        restarts += 1
        # Occasionally use a random seed block to break symmetry more aggressively
        seed = None
        if restarts % 5 == 0:
            # Pick a random block to start with
            seed = [random.randint(0, n_k - 1)]
            
        cover = greedy_cover(seed_blocks=seed, random_tie_break=True)
        cover = remove_redundant(cover)
        
        if len(cover) < best_size:
            best_size = len(cover)
            best_cover = cover
            # Optional: Try to improve best_cover with local search?
            # For now, just keep the best found.
            
        # Check time
        if time.time() > deadline - 5.0:
            break

    # Convert indices back to blocks
    if best_cover is None:
        # Fallback: return empty or single block (should not happen)
        return []
        
    return [list(k_subs[idx]) for idx in best_cover]
