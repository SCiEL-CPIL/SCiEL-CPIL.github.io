import itertools
import time
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110
    
    # 1. Precomputation
    t_subs = list(itertools.combinations(range(v), t))
    t_sub_to_idx = {sub: i for i, sub in enumerate(t_subs)}
    num_t_subs = len(t_subs)
    
    k_blocks = list(itertools.combinations(range(v), k))
    num_k_blocks = len(k_blocks)
    
    # M: num_k_blocks x num_t_subs boolean matrix
    # M[i, j] = 1 if k_blocks[i] contains t_subs[j]
    # To build this efficiently:
    # For each block i, find its t-subsets.
    # block_t_indices[i] = list of j such that t_subs[j] is in k_blocks[i]
    
    block_t_indices = []
    for block in k_blocks:
        indices = []
        for sub in itertools.combinations(block, t):
            indices.append(t_sub_to_idx[sub])
        block_t_indices.append(indices)
        
    # Convert to a sparse matrix or just use the lists.
    # For numpy dot product, we need a dense or sparse matrix.
    # 50k x 3876 is ~200M booleans = 200MB. Too much?
    # 50000 * 3876 * 1 byte = 193 MB.
    # This is acceptable in memory, but creating it might take time.
    # Let's use scipy.sparse if available, or just stick to lists if numpy is too slow to build.
    # Actually, 193MB is fine.
    
    M = np.zeros((num_k_blocks, num_t_subs), dtype=np.int8)
    for i, indices in enumerate(block_t_indices):
        for j in indices:
            M[i, j] = 1
            
    # 2. Greedy Construction
    uncovered = np.ones(num_t_subs, dtype=np.int8)
    solution = []
    
    while np.sum(uncovered) > 0 and time.time() < deadline - 30:
        # Calculate gains
        gains = M.dot(uncovered)
        best_i = np.argmax(gains)
        
        if gains[best_i] == 0:
            break
            
        solution.append(best_i)
        # Update uncovered
        # M[best_i, :] is 1 where covered
        uncovered[M[best_i, :] == 1] = 0
        
    # 3. Local Search (Hill Climbing)
    # cover_count: number of blocks in solution covering each t-subset
    cover_count = np.zeros(num_t_subs, dtype=np.int16)
    for i in solution:
        cover_count[M[i, :] == 1] += 1
        
    improved = True
    max_iters = 100
    iter_count = 0
    
    while improved and iter_count < max_iters and time.time() < deadline - 5:
        improved = False
        iter_count += 1
        
        # Try to remove redundant blocks
        for i in list(solution):
            # Check if any t-subset in block i has cover_count == 1
            # If all > 1, we can remove it.
            counts_in_block = cover_count[M[i, :] == 1]
            if np.min(counts_in_block) > 1:
                # Remove
                cover_count[M[i, :] == 1] -= 1
                solution.remove(i)
                improved = True
                break # Restart scan
                
        if improved:
            continue
            
        # Try swaps
        # For each block i, if it has "critical" t-subsets (count == 1),
        # we must replace it with a block that covers those critical subsets.
        for i in list(solution):
            indices_i = block_t_indices[i]
            criticals = [j for j in indices_i if cover_count[j] == 1]
            
            if not criticals:
                continue # Should have been removed, but maybe race condition?
                
            # Find candidates that cover all criticals
            # Intersect lists of blocks containing each critical
            # To speed up, pick the critical with the fewest blocks containing it
            criticals_sorted = sorted(criticals, key=lambda j: np.sum(M[:, j]))
            if not criticals_sorted:
                continue
            
            # Start with blocks containing the first critical
            cand_blocks = np.where(M[:, criticals_sorted[0]] == 1)[0]
            
            for c in criticals_sorted[1:]:
                cand_blocks = np.intersect1d(cand_blocks, np.where(M[:, c] == 1)[0])
                if len(cand_blocks) == 0:
                    break
                    
            if len(cand_blocks) == 0:
                continue
                
            # Filter out current solution blocks (except i itself, but we are replacing i)
            sol_set = set(solution)
            valid_cands = [c for c in cand_blocks if c not in sol_set]
            
            if not valid_cands:
                continue
                
            # Pick the best candidate
            # Score: number of t-subsets covered by candidate that are currently covered by 1 block (including i)
            # Or simply: maximize total coverage?
            # Let's pick the one that covers the most "low count" subsets.
            best_cand = -1
            best_score = -1
            for c in valid_cands:
                # Score = sum of 1/(cover_count[j]+1) for j in block c?
                # Simpler: count how many subsets in c have cover_count <= 1
                # This helps in making subsets redundant in the future
                score = 0
                for j in block_t_indices[c]:
                    if cover_count[j] <= 1:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_cand = c
                    
            if best_cand != -1:
                # Perform swap
                # Remove i
                cover_count[M[i, :] == 1] -= 1
                # Add best_cand
                cover_count[M[best_cand, :] == 1] += 1
                solution.remove(i)
                solution.append(best_cand)
                improved = True
                break # Restart scan

    # Convert indices back to blocks
    return [list(k_blocks[i]) for i in solution]
