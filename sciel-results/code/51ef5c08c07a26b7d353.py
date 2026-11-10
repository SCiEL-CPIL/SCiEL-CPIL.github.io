import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 580.0
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Lifted vectors: [x, y, z, x^2+y^2+z^2, 1]
    lifted_grid = np.column_stack([grid, (grid**2).sum(axis=1), np.ones(num_points)]).astype(np.float64)
    
    def can_add(p_idx, S_indices):
        """
        Returns True if p_idx can be added to S_indices (no conflict).
        """
        m = len(S_indices)
        if m < 4:
            return True
            
        p_vec = lifted_grid[p_idx]
        
        # Get all combinations of 4 indices from S_indices
        # Using itertools.combinations is okay for generating the indices,
        # but we need to vectorize the determinant part.
        
        # For very large m, this memory usage might be high.
        # If m=50, C(50,4) = 230300.
        # Matrix size: 230300 * 5 * 5 * 8 bytes = 46 MB.
        # This is acceptable.
        
        # Generate indices
        # We need to map S_indices (values) to their positions in the S_vecs array?
        # No, S_indices are indices into lifted_grid.
        # So we can just use them directly to index lifted_grid.
        
        # To vectorize, we need an array of shape (N_combos, 4) containing the indices of the 4 points.
        # Let's use itertools to get the tuples, then stack them.
        # This Python loop to generate the index array is fast compared to the det calculation.
        
        combos = np.array(list(itertools.combinations(S_indices, 4)))
        
        if len(combos) == 0:
            return True
            
        # Index lifted_grid with combos
        # S_vecs_batch has shape (N_combos, 4, 5)
        S_vecs_batch = lifted_grid[combos]
        
        # We need to form matrices of shape (N_combos, 5, 5)
        # Row 0: p_vec (broadcast)
        # Rows 1-4: S_vecs_batch
        
        # Create the batch of matrices
        # Initialize
        M = np.empty((len(combos), 5, 5), dtype=np.float64)
        M[:, 0, :] = p_vec
        M[:, 1:, :] = S_vecs_batch
        
        # Compute determinants
        dets = np.linalg.det(M)
        
        # Check if any determinant is small
        # The problem states |det| < 0.5 is invalid.
        # Since inputs are integers, det is integer. So det == 0.
        # But due to floating point, we use 0.5 threshold.
        
        if np.any(np.abs(dets) < 0.5):
            return False
            
        return True

    best_set_indices = []
    num_restarts = 1000 # Will be limited by time
    
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
            
        order = np.random.permutation(num_points)
        current_indices = []
        
        for p_idx in order:
            if time.time() > deadline:
                break
                
            if can_add(p_idx, current_indices):
                current_indices.append(p_idx)
                
        if len(current_indices) > len(best_set_indices):
            best_set_indices = current_indices
            
    if best_set_indices:
        return grid[best_set_indices].astype(np.int64)
    else:
        return np.empty((0, 3), dtype=np.int64)
