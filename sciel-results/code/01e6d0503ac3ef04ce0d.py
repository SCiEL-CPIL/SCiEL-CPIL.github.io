import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    sq_norms = np.sum(grid**2, axis=1).astype(np.float64)
    num_points = len(grid)
    
    start_time = time.time()
    deadline = 550.0
    
    best_set = []
    best_size = 0
    
    def is_valid_addition(S_indices, p_idx):
        m = len(S_indices)
        if m < 4:
            return True
            
        p = grid[p_idx]
        p_sq = sq_norms[p_idx]
        
        Q = grid[S_indices]
        Q_sq = sq_norms[S_indices]
        
        # If m is small, we can just check.
        # Generate combinations
        # To speed up, we can precompute combinations for a fixed m? 
        # But m changes.
        
        # Use itertools.combinations
        # For m=50, this is ~230k tuples.
        # Converting to numpy array is the cost.
        
        combos = list(itertools.combinations(range(m), 4))
        if not combos:
            return True
            
        combo_arr = np.array(combos, dtype=np.int32)
        
        # Chunked processing to avoid memory spikes if m is large
        chunk_size = 20000
        for i in range(0, len(combo_arr), chunk_size):
            chunk = combo_arr[i:i+chunk_size]
            # (chunk_size, 4, 4)
            M = np.empty((len(chunk), 4, 4), dtype=np.float64)
            M[:, :, 0:3] = Q[chunk] - p
            M[:, :, 3] = Q_sq[chunk] - p_sq
            
            dets = np.linalg.det(M)
            if np.any(np.abs(dets) < 0.5):
                return False
                
        return True

    np.random.seed(42)
    
    # We will run restarts until time is up
    # To make it more effective, we can bias the random order?
    # Uniform random is a good start.
    
    while time.time() < deadline:
        order = np.random.permutation(num_points)
        current_set = []
        
        for p_idx in order:
            if time.time() > deadline:
                break
            
            if is_valid_addition(current_set, p_idx):
                current_set.append(p_idx)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set[:]
            
    if not best_set:
        return np.array([], dtype=np.int64).reshape(0, 3)
        
    return grid[best_set]
