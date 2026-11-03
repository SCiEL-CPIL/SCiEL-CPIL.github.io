import itertools
import numpy as np
import time

def construct(v, k, t):
    start_time = time.time()
    deadline = 115.0 # Leave 5s margin for 120s limit
    
    # Precompute all t-subsets as bitmasks
    subsets = list(itertools.combinations(range(v), t))
    n_subsets = len(subsets)
    
    subset_masks = np.zeros(n_subsets, dtype=np.uint32)
    for i, s in enumerate(subsets):
        mask = 0
        for p in s:
            mask |= (1 << p)
        subset_masks[i] = mask
        
    all_points = list(range(v))
    num_add = k - t
    
    best_blocks = None
    best_len = float('inf')
    
    # Run multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
            
        uncovered = np.ones(n_subsets, dtype=bool)
        blocks = []
        
        while np.any(uncovered):
            if time.time() > deadline:
                break
                
            # Pick a random uncovered subset
            unc_idx = np.where(uncovered)[0]
            if len(unc_idx) == 0:
                break
                
            idx = unc_idx[np.random.randint(len(unc_idx))]
            s0_mask = subset_masks[idx]
            s0_points = subsets[idx]
            s0_set = set(s0_points)
            
            # Generate candidates: supersets of s0 of size k
            remaining = [p for p in all_points if p not in s0_set]
            
            if num_add == 0:
                candidates = [()]
            elif num_add == 1:
                candidates = [(p,) for p in remaining]
            elif num_add == 2:
                candidates = list(itertools.combinations(remaining, 2))
            elif num_add == 3:
                candidates = list(itertools.combinations(remaining, 3))
            else:
                candidates = list(itertools.combinations(remaining, num_add))
                
            # Build masks for candidates
            cand_masks = np.zeros(len(candidates), dtype=np.uint32)
            for i, c in enumerate(candidates):
                mask = s0_mask
                for p in c:
                    mask |= (1 << p)
                cand_masks[i] = mask
            
            # Calculate gains: number of uncovered subsets covered by each candidate
            # covered_mask[i, j] is True if subset j is covered by candidate i
            covered_mask = (subset_masks[None, :] & ~cand_masks[:, None]) == 0
            gains = (uncovered[None, :] & covered_mask).sum(axis=1)
            
            best_idx = np.argmax(gains)
            best_mask = cand_masks[best_idx]
            best_cand = candidates[best_idx]
            
            block_points = list(s0_points) + list(best_cand)
            block_points.sort()
            blocks.append(block_points)
            
            # Update uncovered
            new_covered = (subset_masks & ~best_mask) == 0
            uncovered &= ~new_covered
            
        if len(blocks) < best_len:
            best_blocks = blocks
            best_len = len(blocks)
            
    if best_blocks is None:
        return []
        
    return best_blocks
