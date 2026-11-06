import time
from itertools import combinations
import random

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
    
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    
    # Precompute pairwise intersection sizes
    # inter[i][j] = popcount(mask_i & mask_j)
    # Using a list of bytearrays or lists of ints
    # 1820 * 1820 = 3,312,400 entries. 
    # Python lists of lists are slow. Let's use a flat list or numpy array?
    # Standard library only? "Python standard library, NumPy and SciPy are available."
    # Let's use NumPy for speed.
    
    import numpy as np
    
    masks_np = np.array(masks, dtype=np.uint32)
    
    # Compute pairwise intersections
    # This is O(N^2) bitwise AND and popcount.
    # Vectorized: 
    # We can do this in chunks or just use a loop if N is small enough.
    # 1820^2 = 3.3M ops. A simple loop in Python might take a few seconds.
    # Let's try to vectorize or use a fast method.
    
    # Method: 
    # inter[i][j] = popcount(masks[i] & masks[j])
    
    # Since we can use NumPy:
    # Create a matrix of masks? 1820x1820 uint32 = ~13MB. Feasible.
    # But bitwise AND of two 2D arrays? 
    # masks_np[:, None] & masks_np[None, :] -> 1820x1820 array.
    # Then popcount.
    # NumPy doesn't have a direct popcount for uint32 arrays in all versions, 
    # but we can use `np.bitwise_count` if available (newer numpy) or lookups.
    # Lookup table for popcount of 16-bit chunks?
    # Or just `bin(x).count('1')` in a loop? 3.3M iterations in Python is ~1-2s. Acceptable.
    
    inter = np.zeros((N, N), dtype=np.uint8)
    for i in range(N):
        mi = masks[i]
        for j in range(i, N):
            mj = masks[j]
            v = (mi & mj).bit_count() # Python 3.10+
            inter[i][j] = v
            inter[j][i] = v
            
    # Greedy with randomization
    random.seed(42)
    best_family = []
    best_size = 0
    
    # Number of trials depends on time
    # Each trial is O(N * K^2) roughly, but with the bucket optimization, it's faster.
    # Let's estimate time per trial.
    # N=1820 candidates.
    # For each candidate, check against K chosen.
    # Bucket check: 
    #   Compute inter with C: K ops.
    #   Bucket: K ops.
    #   Check pairs in buckets: Sum |b_s|^2.
    # If K=100, avg |b_s| ~ 25, sum ~ 2500.
    # Total ops per trial ~ 1820 * 2500 = 4.5M ops.
    # In Python, 4.5M simple ops is ~0.5s.
    # So we can run ~100-200 trials in 80s.
    
    num_trials = 200
    
    for trial in range(num_trials):
        if time.time() > deadline:
            break
            
        # Shuffle indices
        indices = list(range(N))
        random.shuffle(indices)
        
        chosen_indices = []
        chosen_masks = [] # redundant if we have indices, but keep for clarity
        
        # For fast access to chosen indices in the bucket logic
        # We just use the global indices.
        
        for idx in indices:
            if time.time() > deadline:
                break
                
            # Check conflict
            conflict = False
            
            # 1. Compute intersection sizes with all chosen
            # inter_chosen = [inter[idx][c] for c in chosen_indices]
            
            # 2. Bucket chosen indices by intersection size with idx
            # buckets = {0: [], 1: [], 2: [], 3: []}
            buckets = [[] for _ in range(4)]
            
            for c_idx in chosen_indices:
                s = inter[idx][c_idx]
                buckets[s].append(c_idx)
                
            # 3. Check for sunflower
            # For each s, check if any pair in buckets[s] has inter[i][j] == s
            for s in range(4):
                b = buckets[s]
                len_b = len(b)
                if len_b < 2:
                    continue
                # Check pairs
                # Optimization: if len_b is large, we might need to be careful.
                # But max intersection size is 3.
                for i in range(len_b):
                    i_idx = b[i]
                    for j in range(i+1, len_b):
                        j_idx = b[j]
                        if inter[i_idx][j_idx] == s:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen_indices.append(idx)
                
        current_size = len(chosen_indices)
        if current_size > best_size:
            best_size = current_size
            best_family = [all_sets[i][1] for i in chosen_indices]
            
    return [sorted(s) for s in best_family]
