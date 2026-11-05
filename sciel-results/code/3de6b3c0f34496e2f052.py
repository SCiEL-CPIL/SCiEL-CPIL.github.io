import itertools
import random
import math
import time
import numpy as np

def construct(v, k, t):
    # Parameters
    points = list(range(v))
    
    # Precompute all subsets of size t
    all_subsets = list(itertools.combinations(points, t))
    n_subsets = len(all_subsets)
    subset_to_idx = {s: i for i, s in enumerate(all_subsets)}
    
    # Precompute all blocks of size k
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    # Build coverage matrix M: M[i, j] = 1 if block i covers subset j
    # To save memory and time, we can represent coverage as bitsets or just use sparse logic.
    # n_blocks ~ 346k, n_subsets ~ 2024.
    # A dense boolean matrix 346k x 2024 is ~ 700 MB. Too big?
    # 346104 * 2024 bytes = ~700MB. This might be too large for some environments, or slow to initialize.
    # Let's use a list of sets of subset indices for each block.
    
    block_subsets = []
    for b in all_blocks:
        subs = [subset_to_idx[s] for s in itertools.combinations(b, t)]
        block_subsets.append(subs)
    
    # Greedy Set Cover
    uncovered = np.ones(n_subsets, dtype=bool)
    selected_blocks = []
    selected_indices = set()
    
    # Precompute gains? No, gains change as uncovered changes.
    # We need to efficiently find the block with max new coverage.
    
    # Optimization: 
    # Instead of iterating all blocks and summing uncovered for each (O(N_blocks * N_subs_per_block)),
    # we can maintain an array `gain` of length n_blocks.
    # Initially, gain[i] = len(block_subsets[i]).
    # When we select a block `b_idx`, we need to decrease the gain of all other blocks `j` 
    # by the number of subsets in `block_subsets[j]` that are also in `block_subsets[b_idx]` AND were uncovered.
    # This is still expensive if done naively.
    
    # Given the small size of n_subsets (2024), we can represent `uncovered` as a numpy array of ints (0/1).
    # For each block i, gain[i] = sum(uncovered[j] for j in block_subsets[i]).
    # Updating gain for all i after selecting b is O(N_blocks * N_subs_per_block).
    # 346k * 35 (avg subsets per block) ~ 12 million ops per step.
    # Number of steps ~ 76.
    # 76 * 12M ~ 900M ops. This might be slow in Python loops.
    
    # Let's use NumPy for the update.
    # Create a matrix `Cover` where Cover[i, j] = 1 if block i covers subset j.
    # Memory: 346104 * 2024 * 1 byte = 700 MB. 
    # If memory is an issue, we can use a sparse representation or just accept the time.
    # Let's try to be smarter.
    
    # Alternative: 
    # Keep `uncovered` as a set of indices.
    # For each block, calculate gain by checking intersection with `uncovered`.
    # This is slow.
    
    # Let's try the NumPy matrix approach but with uint8.
    # If it fails, we fall back to a faster greedy that doesn't check ALL blocks, but a random sample or a heuristic subset.
    # However, for v=24, checking all is best for quality.
    
    try:
        # Initialize coverage matrix
        # To save memory, we can store it as a list of arrays of indices, and use a vectorized trick?
        # Actually, let's just iterate blocks in Python but optimize the inner loop.
        
        # Better approach: 
        # Use a dictionary `subset_count` to track how many selected blocks cover each subset.
        # This allows us to identify redundant blocks easily.
        
        # For Greedy, we need max gain.
        # Let's stick to the standard greedy but optimize the "gain" calculation.
        
        # We will precompute for each block, the list of subset indices it covers.
        # `block_subs` is a list of lists.
        
        # State:
        # `uncovered_mask`: numpy array of bools, size n_subsets.
        # `block_gains`: numpy array of ints, size n_blocks.
        
        # Initialize gains
        block_gains = np.array([len(s) for s in block_subsets], dtype=np.int32)
        
        # We need to update block_gains when we pick a block.
        # If we pick block `b`, the subsets it covers are `S_b`.
        # For any other block `j`, its new gain is `old_gain[j] - |S_j \cap S_b \cap Uncovered|`.
        # This is hard to vectorize without the full matrix.
        
        # Let's use the full matrix if possible.
        # 700MB might be tight. Let's check if we can use a smaller dtype or sparse.
        # Actually, for v=24, k=7, t=3, this is a very specific small problem.
        # Let's try a different heuristic: **Greedy with Randomized Tie-Breaking and Local Search**.
        
        # I will implement a faster greedy that doesn't use the full matrix, but uses a "bucket" approach or just iterates.
        # Wait, 346k blocks * 76 steps * (cost to calc gain).
        # If cost to calc gain is O(35) (number of subsets in a block), total ops = 346k * 76 * 35 = 923M.
        # In Python, 900M simple ops takes ~10-30 seconds. This is feasible!
        
        # So, simple Python loop for greedy is fine.
        
        uncovered_set = set(range(n_subsets))
        selected_block_indices = []
        
        # Precompute block_subsets as a list of lists of ints
        # Already done above as `block_subsets`
        
        # To speed up gain calculation, we can just iterate `block_subsets[i]` and check `in uncovered_set`.
        # `in` for a set is O(1).
        
        while uncovered_set:
            best_gain = -1
            best_block_idx = -1
            
            # Iterate all blocks
            for i, subs in enumerate(block_subsets):
                # Calculate gain: number of subsets in subs that are in uncovered_set
                gain = 0
                # Optimization: if len(subs) <= best_gain, we can't beat it? 
                # No, because some might be uncovered.
                # But if the maximum possible gain (len(subs)) is <= best_gain, skip.
                if len(subs) <= best_gain:
                    continue
                
                for s_idx in subs:
                    if s_idx in uncovered_set:
                        gain += 1
                        if gain > best_gain:
                            # Early termination for this block? No, we need to finish to know if it's the best?
                            # Actually, if gain > best_gain, we update. We can break early if we want to just find A good block, 
                            # but we want THE best.
                            # However, if we break early, we might miss a higher gain later in the list.
                            # So we must check all.
                            pass
                
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            
            if best_block_idx == -1:
                # Should not happen if uncovered_set is not empty and we have all blocks
                break
            
            # Select the block
            selected_block_indices.append(best_block_idx)
            
            # Update uncovered_set
            for s_idx in block_subsets[best_block_idx]:
                if s_idx in uncovered_set:
                    uncovered_set.discard(s_idx)
            
            # Optional: Check if we can stop early if we have enough blocks? No, we need valid cover.
            
        # Convert indices to blocks
        blocks = [list(all_blocks[i]) for i in selected_block_indices]
        
        # Local Search: Try to remove redundant blocks
        # Calculate coverage count for each subset
        subset_coverage = [0] * n_subsets
        for i in selected_block_indices:
            for s_idx in block_subsets[i]:
                subset_coverage[s_idx] += 1
        
        # Remove redundant blocks
        changed = True
        while changed:
            changed = False
            for i in range(len(selected_block_indices)):
                # Check if removing block i makes any subset uncovered
                b_idx = selected_block_indices[i]
                redundant = True
                for s_idx in block_subsets[b_idx]:
                    if subset_coverage[s_idx] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove block i
                    for s_idx in block_subsets[b_idx]:
                        subset_coverage[s_idx] -= 1
                    selected_block_indices.pop(i)
                    changed = True
                    break # Restart loop as indices changed
        
        blocks = [list(all_blocks[i]) for i in selected_block_indices]
        
        # Further Local Search: Try to swap points in blocks to potentially allow removal of other blocks
        # This is complex. Given time, the greedy + redundancy removal is a solid baseline.
        # The best known is 76. Greedy usually gets close.
        
        return blocks
        
    except Exception as e:
        # Fallback to a simple random/greedy if memory or time issues
        # Just return a valid cover, even if not optimal
        # Simple greedy without optimization
        uncovered_set = set(range(n_subsets))
        selected_block_indices = []
        while uncovered_set:
            best_gain = -1
            best_block_idx = -1
            for i, subs in enumerate(block_subsets):
                if len(subs) <= best_gain:
                    continue
                gain = sum(1 for s in subs if s in uncovered_set)
                if gain > best_gain:
                    best_gain = gain
                    best_block_idx = i
            if best_block_idx == -1: break
            selected_block_indices.append(best_block_idx)
            for s_idx in block_subsets[best_block_idx]:
                if s_idx in uncovered_set:
                    uncovered_set.discard(s_idx)
        
        blocks = [list(all_blocks[i]) for i in selected_block_indices]
        return blocks
