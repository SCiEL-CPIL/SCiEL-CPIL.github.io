import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for the entire grid
    sq_norms_grid = (grid ** 2).sum(axis=1)

    def can_add(arr: np.ndarray, p: np.ndarray, sq_p: float) -> bool:
        m = len(arr)
        if m < 4:
            return True
        
        # W rows: (arr_i - p, |arr_i|^2 - |p|^2)
        # We need to check if any 4 rows are linearly dependent.
        
        # Compute differences and squared norm differences
        diff = arr - p
        sq_diff = (arr ** 2).sum(axis=1) - sq_p
        
        # If m is 4, just check the single determinant
        if m == 4:
            W = np.column_stack((diff, sq_diff))
            det = np.linalg.det(W)
            return abs(det) >= 0.5

        # For m > 4, we need to check all combinations of 4 rows.
        # To avoid constructing a huge list of combinations every time,
        # we can generate indices.
        # However, constructing the matrix W is cheap.
        # The expensive part is the determinants.
        
        # Optimization: If m is large, we can check subsets.
        # But m is small (max ~30-40).
        
        # Generate combinations of indices
        # Using itertools is fine for m < 50.
        indices = list(itertools.combinations(range(m), 4))
        
        # Extract the relevant rows
        # This creates a view or copy? W[indices] creates a copy.
        # W is (m, 4). indices is (C, 4).
        # sub_mats is (C, 4, 4).
        
        W = np.column_stack((diff, sq_diff))
        
        # If m is very large, this might be slow.
        # But given the constraint, m won't be huge.
        
        sub_mats = W[indices]
        
        # Batch determinant
        # np.linalg.det can handle (..., M, M)
        dets = np.linalg.det(sub_mats)
        
        # Check if any determinant is close to 0
        return not np.any(np.abs(dets) < 0.5)

    best_set = np.empty((0, 3), dtype=np.int64)
    start_time = time.time()
    time_limit = 590.0 # 10s margin
    
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Check time before starting a new restart to avoid overshooting
        if time.time() - start_time > time_limit - 2:
            break
            
        order = rng.permutation(num_points)
        current_set = np.empty((0, 3), dtype=np.int64)
        
        for idx in order:
            # Pruning: if we can't beat the best, stop
            # remaining points = num_points - (current index in order)
            # But we don't know current index easily in this loop structure without enumerate.
            # Let's use enumerate.
            pass
        
        # Redo loop with enumerate for pruning
        current_set = np.empty((0, 3), dtype=np.int64)
        for i, idx in enumerate(order):
            # If current size + remaining <= best size, break
            if len(current_set) + (num_points - i) <= len(best_set):
                break
                
            p = grid[idx]
            sq_p = sq_norms_grid[idx]
            
            if can_add(current_set, p, sq_p):
                if len(current_set) == 0:
                    current_set = p[None, :]
                else:
                    current_set = np.vstack([current_set, p])
        
        if len(current_set) > len(best_set):
            best_set = current_set.copy()
            
    return best_set.astype(np.int64)
