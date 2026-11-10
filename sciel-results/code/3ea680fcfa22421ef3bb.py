import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for the entire grid
    sq_norms_grid = (grid ** 2).sum(axis=1)

    def can_add(buf: np.ndarray, sq_buf: np.ndarray, m: int, p: np.ndarray, sq_p: float) -> bool:
        if m < 4:
            return True
        
        # W rows: (arr_i - p, |arr_i|^2 - |p|^2)
        # We need to check if any 4 rows are linearly dependent.
        
        # Compute differences and squared norm differences
        arr = buf[:m]
        sq_arr = sq_buf[:m]
        
        diff = arr - p
        sq_diff = sq_arr - sq_p
        
        # If m is 4, just check the single determinant
        if m == 4:
            W = np.column_stack((diff, sq_diff))
            det = np.linalg.det(W)
            return abs(det) >= 0.5

        # Generate combinations of indices
        indices = list(itertools.combinations(range(m), 4))
        
        # Extract the relevant rows
        W = np.column_stack((diff, sq_diff))
        
        sub_mats = W[indices]
        
        # Batch determinant
        dets = np.linalg.det(sub_mats)
        
        # Check if any determinant is close to 0
        return not np.any(np.abs(dets) < 0.5)

    best_set = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 620.0 # Use more of the available time
    
    rng = np.random.default_rng(42)
    
    # Pre-allocate buffers for the greedy search
    max_k = num_points
    buffer = np.empty((max_k, 3), dtype=np.int64)
    sq_buffer = np.empty(max_k, dtype=np.int64)
    
    while time.time() - start_time < time_limit:
        # Check time before starting a new restart to avoid overshooting
        if time.time() - start_time > time_limit - 2:
            break
            
        order = rng.permutation(num_points)
        m = 0
        
        for i, idx in enumerate(order):
            # If current size + remaining <= best size, break
            if m + (num_points - i) <= len(best_set):
                break
                
            p = grid[idx]
            sq_p = sq_norms_grid[idx]
            
            if can_add(buffer, sq_buffer, m, p, sq_p):
                buffer[m] = p
                sq_buffer[m] = sq_p
                m += 1
        
        if m > len(best_set):
            best_set = buffer[:m].copy()
            
    return best_set.astype(np.int64)
