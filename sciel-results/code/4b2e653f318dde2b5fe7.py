# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical.
    Uses multi-start greedy + local search (add and swap moves) with vectorized determinant checks.
    """
    n = 8
    deadline = time.time() + 580  # 20s safety margin
    
    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point
    
    # Pre-allocate matrix for determinant calculation
    M = np.empty((4, 4), dtype=np.float64)

    def can_add_to_set(arr_indices: np.ndarray, new_idx: int) -> bool:
        """Check if adding new_idx to the set keeps it valid."""
        m = len(arr_indices)
        if m < 4:
            return True
        
        p = grid[new_idx]
        pi_sq = sq[new_idx]
        
        # Get points and squared norms of existing set
        S_pts = grid[arr_indices]
        S_sqs = sq[arr_indices]
        
        # Iterate over all 4-combinations of the existing set
        # Using itertools.combinations
        for combo in itertools.combinations(range(m), 4):
            i1, i2, i3, i4 = combo
            
            # Fill matrix M
            # Row 1
            M[0, 0] = S_pts[i1, 0] - p[0]
            M[0, 1] = S_pts[i1, 1] - p[1]
            M[0, 2] = S_pts[i1, 2] - p[2]
            M[0, 3] = S_sqs[i1] - pi_sq
            
            # Row 2
            M[1, 0] = S_pts[i2, 0] - p[0]
            M[1, 1] = S_pts[i2, 1] - p[1]
            M[1, 2] = S_pts[i2, 2] - p[2]
            M[1, 3] = S_sqs[i2] - pi_sq
            
            # Row 3
            M[2, 0] = S_pts[i3, 0] - p[0]
            M[2, 1] = S_pts[i3, 1] - p[1]
            M[2, 2] = S_pts[i3, 2] - p[2]
            M[2, 3] = S_sqs[i3] - pi_sq
            
            # Row 4
            M[3, 0] = S_pts[i4, 0] - p[0]
            M[3, 1] = S_pts[i4, 1] - p[1]
            M[3, 2] = S_pts[i4, 2] - p[2]
            M[3, 3] = S_sqs[i4] - pi_sq
            
            if abs(np.linalg.det(M)) < 0.5:
                return False
        return True

    def greedy_build(order: np.ndarray) -> np.ndarray:
        """Greedy: add points in given order if feasible."""
        arr = np.empty(0, dtype=np.int64)
        arr_set = set()
        for idx in order:
            if idx in arr_set:
                continue
            if can_add_to_set(arr, idx):
                arr = np.append(arr, idx)
                arr_set.add(idx)
        return arr

    def local_search(arr: np.ndarray) -> np.ndarray:
        """Local search with Add and Swap moves."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Phase 1: Try to add points
            # Iterate over complement
            for idx in comp:
                if time.time() > deadline:
                    break
                if can_add_to_set(current, idx):
                    current = np.append(current, idx)
                    current_set.add(idx)
                    comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                    improved = True
                    break # Restart add phase
            
            if improved:
                continue
                
            # Phase 2: Try swaps
            # Try each removal
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without = np.delete(current, i)
                
                # Try adding each complement point
                for j in range(len(comp)):
                    if time.time() > deadline:
                        break
                    add_idx = comp[j]
                    if can_add_to_set(without, add_idx):
                        # Valid swap
                        current = np.concatenate([without, [add_idx]])
                        current_set = set(current.tolist())
                        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                        improved = True
                        break # Restart swap phase
                if improved:
                    break
                    
        return current

    rng = np.random.RandomState(42)
    best = None
    best_len = 0
    
    # Multi-start
    # We will run multiple restarts, each consisting of greedy + local search
    # Time budget: 580s
    
    while time.time() < deadline - 10:
        order = rng.permutation(N)
        arr = greedy_build(order)
        arr = local_search(arr)
        
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
            
        # If we have a large set, we might spend more time on local search
        # But the loop above handles it.
        
    if best is None:
        best = np.array([0, 1, 2, 3], dtype=np.int64)
        
    return grid[best].astype(np.int64)

# EVOLVE-BLOCK-END
