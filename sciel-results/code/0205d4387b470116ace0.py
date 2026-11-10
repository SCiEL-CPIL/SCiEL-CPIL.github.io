# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid {0,...,7}^3
    such that no five of the selected points are co-spherical (or co-planar).
    
    Uses vectorized determinant checks for speed, multi-restart greedy with
    diverse orderings, and swap-based local improvement.
    """
    n = 8
    deadline = time.time() + 580  # safety margin before 630s kill
    
    # Precompute grid
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    grid_sq = (grid ** 2).sum(axis=1, dtype=np.float64)
    N = len(grid)  # 512
    
    # Precompute all C(m,4) index combinations dynamically as m grows
    # For efficiency, we'll compute them on the fly but vectorize the det check
    
    def vectorized_can_add(arr, arr_sq, p, p_sq):
        """
        Check if point p can be added to arr without creating 5 co-spherical points.
        Vectorized: compute all C(m,4) determinants at once.
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Build the m x 4 matrix: rows are (qi - p, |qi|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr.astype(np.float64) - p.astype(np.float64)
        w[:, 3] = arr_sq - p_sq
        
        # For each combination of 4 rows, compute determinant
        # Vectorize by processing all combinations at once
        # Generate all C(m,4) combinations
        if m <= 4:
            if m == 4:
                return abs(np.linalg.det(w)) >= 0.5
            return True
        
        # For larger m, we need to check all C(m,4) submatrices
        # Use a chunked approach to avoid memory issues for large m
        combos = list(itertools.combinations(range(m), 4))
        num_combos = len(combos)
        
        if num_combos == 0:
            return True
        
        # Process in chunks to manage memory
        chunk_size = min(10000, num_combos)
        for start in range(0, num_combos, chunk_size):
            end = min(start + chunk_size, num_combos)
            idx = np.array(combos[start:end])  # shape (chunk, 4)
            
            # Extract submatrices: w[idx[:,i]] for i=0..3, stack to (chunk, 4, 4)
            # w[idx] has shape (chunk, 4, 4) where w[idx[c, i]] is row i of combo c
            # Actually: w[idx[c, i]] gives us the row. We need w[idx] where idx is (chunk,4)
            # w[idx] with idx shape (chunk, 4) gives shape (chunk, 4, 4)? No.
            # w has shape (m, 4). w[idx] where idx is (chunk, 4) gives shape (chunk, 4, 4).
            submats = w[idx]  # shape (chunk, 4, 4)
            
            # Compute determinants for all submatrices
            # Use slogdet for numerical stability
            signs, logabs = np.linalg.slogdet(submats)
            dets = signs * np.exp(logabs)
            
            if np.any(np.abs(dets) < 0.5):
                return False
        
        return True
    
    def greedy_fill(order, grid_sq):
        """Greedy fill with given order, using vectorized can_add."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        
        for idx in order:
            p = grid[idx]
            p_sq = grid_sq[idx]
            if vectorized_can_add(arr, arr_sq, p, p_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = p_sq.copy()
                else:
                    arr = np.vstack([arr, p[None, :]])
                    arr_sq = np.append(arr_sq, p_sq)
        
        return arr, arr_sq
    
    def local_improve_swap(arr, arr_sq, grid, grid_sq, deadline):
        """
        Swap-based local improvement: try to remove one point and add one or two others.
        Returns improved arr, arr_sq.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            m = len(arr)
            
            # Build set of indices in arr for quick lookup
            arr_set = set(map(tuple, arr))
            
            # Try single swaps: remove one, add one
            for remove_idx in range(m):
                if time.time() > deadline:
                    return arr, arr_sq
                
                p_remove = arr[remove_idx]
                p_remove_sq = arr_sq[remove_idx]
                
                # Build arr without this point
                if m == 1:
                    arr_temp = np.empty((0, 3), dtype=np.int64)
                    arr_sq_temp = np.empty(0, dtype=np.float64)
                else:
                    mask = np.ones(m, dtype=bool)
                    mask[remove_idx] = False
                    arr_temp = arr[mask]
                    arr_sq_temp = arr_sq[mask]
                
                # Try to add each grid point not in arr_temp
                in_arr_temp = set(map(tuple, arr_temp))
                
                for idx in range(len(grid)):
                    p = grid[idx]
                    if tuple(p) in in_arr_temp:
                        continue
                    p_sq = grid_sq[idx]
                    if vectorized_can_add(arr_temp, arr_sq_temp, p, p_sq):
                        # Found a valid swap
                        if len(arr_temp) == 0:
                            arr = p[None, :]
                            arr_sq = p_sq.copy()
                        else:
                            arr = np.vstack([arr_temp, p[None, :]])
                            arr_sq = np.append(arr_sq_temp, p_sq)
                        improved = True
                        break
                if improved:
                    break
            
            # Try double swaps: remove one, add two (if time permits)
            if not improved and m >= 1 and time.time() < deadline:
                for remove_idx in range(m):
                    if time.time() > deadline:
                        return arr, arr_sq
                    
                    p_remove = arr[remove_idx]
                    p_remove_sq = arr_sq[remove_idx]
                    
                    if m == 1:
                        arr_temp = np.empty((0, 3), dtype=np.int64)
                        arr_sq_temp = np.empty(0, dtype=np.float64)
                    else:
                        mask = np.ones(m, dtype=bool)
                        mask[remove_idx] = False
                        arr_temp = arr[mask]
                        arr_sq_temp = arr_sq[mask]
                    
                    in_arr_temp = set(map(tuple, arr_temp))
                    
                    # Find candidates that can be added to arr_temp
                    candidates = []
                    for idx in range(len(grid)):
                        p = grid[idx]
                        if tuple(p) in in_arr_temp:
                            continue
                        p_sq = grid_sq[idx]
                        if vectorized_can_add(arr_temp, arr_sq_temp, p, p_sq):
                            candidates.append(idx)
                    
                    # Try pairs of candidates
                    for i in range(len(candidates)):
                        if time.time() > deadline:
                            return arr, arr_sq
                        for j in range(i + 1, len(candidates)):
                            p1 = grid[candidates[i]]
                            p2 = grid[candidates[j]]
                            p1_sq = grid_sq[candidates[i]]
                            p2_sq = grid_sq[candidates[j]]
                            
                            # Try adding p1 then p2 to arr_temp
                            if len(arr_temp) == 0:
                                arr_t1 = p1[None, :]
                                arr_sq_t1 = p1_sq.copy()
                            else:
                                arr_t1 = np.vstack([arr_temp, p1[None, :]])
                                arr_sq_t1 = np.append(arr_sq_temp, p1_sq)
                            
                            if vectorized_can_add(arr_t1, arr_sq_t1, p2, p2_sq):
                                # Valid double swap
                                arr = np.vstack([arr_t1, p2[None, :]])
                                arr_sq = np.append(arr_sq_t1, p2_sq)
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
        
        return arr, arr_sq
    
    # Multi-restart greedy with diverse orderings
    np.random.seed(42)
    best = None
    best_len = 0
    
    num_restarts = 500
    restart_count = 0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        restart_count += 1
        
        # Diverse ordering strategies
        strategy = restart % 5
        if strategy == 0:
            order = np.random.permutation(N)
        elif strategy == 1:
            # Sort by squared norm
            order = np.argsort(grid_sq)
            order = order[np.random.permutation(len(order))]
        elif strategy == 2:
            # Sort by x coordinate, then random within
            x_coords = grid[:, 0]
            order = np.lexsort((np.random.permutation(N), x_coords))
        elif strategy == 3:
            # Random but with bias toward corners
            weights = np.ones(N)
            # Boost corners (points with coords in {0,7})
            corner_score = np.sum((grid == 0) | (grid == 7), axis=1)
            weights = 1.0 + corner_score * 0.5
            probs = weights / weights.sum()
            order = np.random.choice(N, size=N, replace=False, p=probs)
        else:
            # Hilbert-like space-filling curve ordering (approximate)
            # Use Morton/Z-order code
            morton = (grid[:, 0] * 64) + (grid[:, 1] * 8) + grid[:, 2]
            order = np.argsort(morton)
            order = order[np.random.permutation(len(order))]
        
        # Greedy fill
        arr, arr_sq = greedy_fill(order, grid_sq)
        
        # Local improvement
        if time.time() < deadline - 30:
            arr, arr_sq = local_improve_swap(arr, arr_sq, grid, grid_sq, deadline)
        
        if best is None or len(arr) > best_len:
            best = arr.copy()
            best_len = len(arr)
    
    # Final verification
    if best is None:
        # Fallback: just take first few grid points
        best = grid[:10]
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
