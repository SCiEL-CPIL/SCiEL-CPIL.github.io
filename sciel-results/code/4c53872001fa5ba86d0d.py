import numpy as np
import itertools
import time

def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 580  # 20s safety margin
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    
    def can_add(arr, p, m=None):
        if m is None:
            m = len(arr)
        if m < 4:
            return True
        # Build the lifted vectors for existing points relative to p
        # w[i] = (arr[i] - p, |arr[i]|^2 - |p|^2)
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        # Check all C(m,4) combinations
        # For performance, early exit if any determinant is ~0
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            det = np.linalg.det(sub)
            if abs(det) < 0.5:
                return False
        return True
    
    def can_add_fast(arr, p, m=None):
        """Vectorized version for speed"""
        if m is None:
            m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        
        # Generate all 4-combinations as indices
        if m <= 4:
            sub = w
            return abs(np.linalg.det(sub)) >= 0.5
        
        # For larger m, we need to check all C(m,4) subsets
        # This is expensive, so let's use a smarter approach
        # Actually, let's just do it naively but with early exit
        combos = list(itertools.combinations(range(m), 4))
        for c in combos:
            sub = w[list(c)]
            det = np.linalg.det(sub)
            if abs(det) < 0.5:
                return False
        return True
    
    best = None
    best_len = 0
    
    def try_greedy(order, start_arr=None):
        nonlocal best, best_len
        arr = start_arr.copy() if start_arr is not None else np.empty((0, 3), dtype=np.int64)
        for p in order:
            if time.time() > deadline:
                break
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                if len(arr) > best_len:
                    best = arr.copy()
                    best_len = len(arr)
        return arr
    
    # Multiple random greedy restarts
    rng = np.random.RandomState(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 30:
            break
        order = grid[rng.permutation(num_points)]
        try_greedy(order)
    
    # Local search: try to improve by swapping
    if best is not None:
        for swap_iter in range(20):
            if time.time() > deadline - 10:
                break
            improved = False
            arr = best.copy()
            m = len(arr)
            # Try removing one point and adding up to two new ones
            for i in range(m):
                if time.time() > deadline - 5:
                    break
                # Remove point i
                arr_without_i = np.delete(arr, i, axis=0)
                # Try to add two new points
                candidates = [p for p in grid if p not in arr_without_i]
                # Try adding one first
                for p in candidates[:50]:  # Limit for speed
                    if can_add(arr_without_i, p):
                        new_arr = np.vstack([arr_without_i, p])
                        if len(new_arr) > m:
                            # Now try to add one more
                            for p2 in candidates[:50]:
                                if np.array_equal(p2, p):
                                    continue
                                if can_add(new_arr, p2):
                                    new_arr2 = np.vstack([new_arr, p2])
                                    if len(new_arr2) > m:
                                        best = new_arr2
                                        best_len = len(new_arr2)
                                        improved = True
                                        break
                        break
            if not improved:
                break
    
    return best.astype(np.int64)
