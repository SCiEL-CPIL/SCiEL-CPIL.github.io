import numpy as np
import itertools
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # 1. Generate all t-subsets
    t_subsets = list(itertools.combinations(range(v), t))
    num_t_subsets = len(t_subsets)
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    
    # 2. Generate all k-subsets (blocks)
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)
    
    # 3. Build the incidence matrix: M[block_idx, t_subset_idx] = 1 if block covers t_subset
    # Using a dense matrix for speed with NumPy dot product
    # Size: ~50000 x 4000 = 200 million entries. 
    # Using uint8 to save memory and speed up operations.
    M = np.zeros((num_k_subsets, num_t_subsets), dtype=np.uint8)
    
    for i, block in enumerate(k_subsets):
        # Find all t-subsets of this block
        # Since k=7, t=4, we iterate over combinations of the block
        for t_sub in itertools.combinations(block, t):
            idx = t_subset_index[t_sub]
            M[i, idx] = 1
            
    # 4. Greedy Set Cover
    uncovered = np.ones(num_t_subsets, dtype=np.uint8)
    blocks = []
    
    # Pre-compute row sums? No, we need dynamic scores based on 'uncovered'.
    # scores = M @ uncovered
    
    while np.any(uncovered):
        if time.time() > deadline:
            break
            
        # Calculate how many uncovered t-subsets each block covers
        # M is (N_blocks, N_t_subsets), uncovered is (N_t_subsets,)
        # Result is (N_blocks,)
        scores = M @ uncovered
        
        # Find the block with the maximum coverage
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        
        if best_score == 0:
            # This shouldn't happen if the problem is solvable and we haven't covered all
            # But if it does, we might be stuck or have a bug.
            # For safety, pick a random block or break.
            break
            
        blocks.append(list(k_subsets[best_idx]))
        
        # Update uncovered: set to 0 all t-subsets covered by this block
        # M[best_idx, :] is a vector of 0s and 1s
        # Where it is 1, set uncovered to 0
        uncovered[M[best_idx] == 1] = 0
        
    # The greedy algorithm is deterministic. 
    # To potentially improve, we could run multiple times with random tie-breaking.
    # But given the time budget, a single high-quality greedy pass with NumPy is very fast.
    # Let's check if we have time for a second pass with a different tie-breaker.
    
    if time.time() < deadline - 5:
        # Try a second pass with random tie-breaking among top candidates
        uncovered2 = np.ones(num_t_subsets, dtype=np.uint8)
        blocks2 = []
        
        while np.any(uncovered2):
            if time.time() > deadline:
                break
                
            scores = M @ uncovered2
            max_score = np.max(scores)
            
            if max_score == 0:
                break
                
            # Find all blocks with max score
            candidates = np.where(scores == max_score)[0]
            
            # Pick one randomly
            best_idx = candidates[np.random.randint(len(candidates))]
            
            blocks2.append(list(k_subsets[best_idx]))
            uncovered2[M[best_idx] == 1] = 0
            
        if len(blocks2) < len(blocks):
            blocks = blocks2
            
    return blocks
