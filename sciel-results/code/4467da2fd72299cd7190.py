import itertools
import time
import random
import sys
import numpy as np

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Precomputation
    # 1. Generate all t-subsets and map them to indices
    t_subsets = list(itertools.combinations(range(v), t))
    t_subset_index = {s: i for i, s in enumerate(t_subsets)}
    num_t_subsets = len(t_subsets)

    # 2. Generate all k-subsets
    k_subsets = list(itertools.combinations(range(v), k))
    num_k_subsets = len(k_subsets)

    # 3. Build cover_lists: for each k-subset, list of t-subset indices it covers
    cover_lists = []
    for ks in k_subsets:
        covered = [t_subset_index[ts] for ts in itertools.combinations(ks, t)]
        cover_lists.append(covered)

    # 4. Build t_to_k: for each t-subset, list of k-subset indices that cover it
    t_to_k = [[] for _ in range(num_t_subsets)]
    for i, covered in enumerate(cover_lists):
        for j in covered:
            t_to_k[j].append(i)

    best_blocks = None
    best_count = float('inf')

    def fast_greedy(rng):
        """
        Fast greedy using restricted candidates.
        """
        uncovered = np.ones(num_t_subsets, dtype=bool)
        current_blocks = []
        
        # To speed up "pick random uncovered", we can maintain a list of uncovered indices
        # But np.where is O(N) which is 11628. That's fast.
        
        while uncovered.any():
            if time.time() > deadline - 5:
                return None, None
            
            # Pick a random uncovered t-subset
            unc_indices = np.where(uncovered)[0]
            if len(unc_indices) == 0:
                break
            u_idx = rng.choice(unc_indices)
            
            # Get candidates covering u_idx
            candidates = t_to_k[u_idx]
            
            # Evaluate gain for each candidate
            best_gain = -1
            best_block_idx = -1
            
            # Optimization: If we have many candidates, we might want to sample?
            # No, 364 is small enough to evaluate all.
            for cb in candidates:
                # Gain is number of currently uncovered t-subsets covered by cb
                # cover_lists[cb] is a list of indices
                gain = np.sum(uncovered[cover_lists[cb]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = cb
                elif gain == best_gain:
                    # Random tie break
                    if rng.random() < 0.5:
                        best_block_idx = cb
            
            if best_block_idx == -1:
                # Should not happen if coverage is complete
                return None, None
                
            current_blocks.append(k_subsets[best_block_idx])
            
            # Update uncovered
            uncovered[cover_lists[best_block_idx]] = False
            
        return current_blocks, np.zeros(num_k_subsets, dtype=bool) # used mask not strictly needed for return, just blocks

    # Run multiple restarts
    num_restarts = 200 # Adjust based on speed
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(restart)
        
        blocks, _ = fast_greedy(rng)
        
        if blocks is not None:
            count = len(blocks)
            if count < best_count:
                best_count = count
                best_blocks = [list(b) for b in blocks]
                
                # Optional: Quick Local Search
                # If time permits, try to improve
                if time.time() < deadline - 5 and count > 300: # Only if we have a decent solution
                    # Simple 1-for-1 swap or removal check
                    # This part is complex to implement safely in the remaining time, 
                    # so we rely on restarts.
                    pass

    # Fallback if no solution found (should not happen)
    if best_blocks is None:
        # Deterministic greedy
        uncovered = np.ones(num_t_subsets, dtype=bool)
        current_blocks = []
        while uncovered.any():
            unc_indices = np.where(uncovered)[0]
            u_idx = unc_indices[0]
            candidates = t_to_k[u_idx]
            best_gain = -1
            best_block_idx = -1
            for cb in candidates:
                gain = np.sum(uncovered[cover_lists[cb]])
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = cb
            current_blocks.append(k_subsets[best_block_idx])
            uncovered[cover_lists[best_block_idx]] = False
        best_blocks = [list(b) for b in current_blocks]

    return best_blocks
