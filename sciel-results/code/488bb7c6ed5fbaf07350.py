import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five selected points are co-spherical (or co-planar).
    """
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    r2 = (grid ** 2).sum(axis=1)
    
    # Precompute combinations for m up to a reasonable limit
    # We will use this to speed up the inner loop
    max_m = 20
    combos_cache = {}
    for m in range(4, max_m + 1):
        combos_cache[m] = list(itertools.combinations(range(m), 4))
        
    start_time = time.time()
    deadline = start_time + 590
    
    best_set = []
    best_len = 0
    
    def can_add(p_idx, S_indices):
        """
        Check if p_idx can be added to the set defined by S_indices.
        S_indices is a list of indices into the grid.
        """
        m = len(S_indices)
        if m < 4:
            return True
        
        p = grid[p_idx]
        rp2 = r2[p_idx]
        
        # Get precomputed combinations for this size m
        # If m is too large, compute on the fly (unlikely to happen often)
        if m in combos_cache:
            combos = combos_cache[m]
        else:
            combos = list(itertools.combinations(range(m), 4))
            
        num_combos = len(combos)
        if num_combos == 0:
            return True
            
        # Prepare the batch matrix
        # Each combo gives a 4x4 matrix.
        # We stack them into (num_combos, 4, 4)
        mats = np.empty((num_combos, 4, 4), dtype=np.float64)
        
        # Fill the matrix
        # We can vectorize the filling partially?
        # S_indices is a list. grid is an array.
        # Let's just loop, it's small.
        for k, combo in enumerate(combos):
            # combo is a tuple of 4 indices into S_indices
            # Get the actual grid indices
            s_idxs = [S_indices[i] for i in combo]
            qs = grid[s_idxs] # (4, 3)
            qr2s = r2[s_idxs] # (4,)
            
            # Rows: q - p, |q|^2 - |p|^2
            # q - p is (4, 3)
            # |q|^2 - |p|^2 is (4,)
            mats[k, :, :3] = qs - p
            mats[k, :, 3] = qr2s - rp2
            
        # Compute determinants
        dets = np.linalg.det(mats)
        
        # Check for singularity
        if np.any(np.abs(dets) < 0.5):
            return False
        return True

    np.random.seed(0)
    
    for restart in range(50000):
        if time.time() >= deadline:
            break
            
        # Random permutation
        order = np.random.permutation(N)
        
        S_indices = []
        
        for p_idx in order:
            # Optional: Early stop if we've reached a very high number
            # But we want max, so keep going.
            if can_add(p_idx, S_indices):
                S_indices.append(p_idx)
                
        if len(S_indices) > best_len:
            best_len = len(S_indices)
            best_set = S_indices.copy()
            
            # If we find a set of size 12 or more, it's a good result.
            # We can continue to search for 13+.
            
    if not best_set:
        # Fallback: return a small valid set
        return grid[:4].astype(np.int64)
        
    return grid[best_set].astype(np.int64)
