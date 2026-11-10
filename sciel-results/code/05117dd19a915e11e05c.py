import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Selects the maximum number of points from an 8x8x8 grid such that no 
    5 points are co-spherical (or co-planar).
    Uses a randomized greedy algorithm with vectorized validity checks
    and multiple restarts within a time limit.
    """
    n = 8
    # Time limit for the function to finish, leaving margin for the evaluator
    deadline = time.time() + 580
    
    # Generate all grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)
    
    # Precompute squared norms for all grid points
    sq_norms_grid = (grid ** 2).sum(axis=1)
    
    def can_add(arr_idx, p_idx):
        """
        Checks if adding grid point p_idx to the set of points defined by indices arr_idx
        violates the no-5-co-spherical condition.
        arr_idx: array of indices into grid
        p_idx: index of candidate point
        """
        m = len(arr_idx)
        if m < 4:
            return True
        
        # Get the actual coordinates
        # arr is (m, 3), p is (3,)
        arr_pts = grid[arr_idx]
        p_pts = grid[p_idx]
        
        # Calculate differences and squared norm differences
        diff = arr_pts - p_pts
        r = sq_norms_grid[arr_idx] - sq_norms_grid[p_idx]
        
        # Build the (m, 4) matrix W
        W = np.column_stack((diff, r))
        
        # Get all combinations of 4 indices from 0 to m-1
        # For large m, this list can be huge. 
        # m is at most ~50-60 in good solutions. C(60,4) ~ 487k.
        # Creating a list of 487k tuples and then an array is slow.
        
        # Optimization: If m is small, just do it.
        # If m is large, we might need a different approach.
        # But C(50,4) is 230k. np.array(list(...)) takes a bit of time.
        
        # Alternative: Use np.triu_indices? No, that's for pairs.
        # We can use a recursive generator or just accept the cost if vectorized det is fast.
        
        # Let's try to generate indices more efficiently if possible.
        # For now, standard itertools is okay if we run many restarts.
        
        # If m is very large, this might be slow.
        # Let's add a check: if m > 60, we might need to be careful.
        # But greedy usually stops adding when it can't anymore.
        
        idx = np.array(list(itertools.combinations(range(m), 4)))
        if len(idx) == 0:
            return True
            
        subs = W[idx]
        dets = np.linalg.det(subs)
        
        return not np.any(np.abs(dets) < 0.5)

    np.random.seed(42)
    best_arr = None
    best_len = 0
    
    # We will run restarts until we hit the deadline
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # Random permutation of grid points
        order = np.random.permutation(N)
        
        current_set = []
        # We can also keep a boolean mask or just list of indices
        for p_idx in order:
            if can_add(np.array(current_set), p_idx):
                current_set.append(p_idx)
                
        if len(current_set) > best_len:
            best_len = len(current_set)
            best_arr = np.array(current_set)
            
        # If we found a very good solution, we might still try to improve
        # but the main gain is from finding a better initial greedy path.
        
    return grid[best_arr] if best_arr is not None else np.empty((0, 3), dtype=np.int64)
