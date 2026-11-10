# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
from math import comb

def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid such that
    no five points are co-spherical (including co-planar).
    
    Uses greedy construction with multiple restarts plus local swap improvement.
    """
    n = 8
    deadline = time.time() + 580  # safety margin before 600s
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    
    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1)
    
    def can_add_fast(arr, p_idx, p, sq_p):
        """Check if adding point p (at index p_idx) to arr keeps the set valid."""
        m = len(arr)
        if m < 4:
            return True
        
        # Build the lifted matrix: rows are (arr - p, |arr|² - |p|²)
        # arr is (m, 3), p is (3,)
        diff = arr - p  # (m, 3)
        sq_diff = sq_norms[arr] - sq_p  # (m,)
        
        # For each combination of 4 existing points, check det of 4x4 matrix
        # Matrix rows: [q_i - p, |q_i|² - |p|²] for i in {1,2,3,4}
        # This is a 4x4 matrix
        
        # Use integer arithmetic for speed and exactness
        # det of 4x4 with integer entries: use np.linalg.det but check |det| >= 0.5
        # Since entries are integers, det is integer, so |det| >= 1 iff valid
        
        indices = np.arange(m)
        # Get all 4-combinations
        combos = list(itertools.combinations(indices, 4))
        if not combos:
            return True
        
        # Vectorize: build all 4x4 matrices at once
        # For each combo (i,j,k,l), matrix is:
        # [arr[i]-p, sq_i - sq_p]
        # [arr[j]-p, sq_j - sq_p]
        # [arr[k]-p, sq_k - sq_p]
        # [arr[l]-p, sq_l - sq_p]
        
        # Build in batches for memory efficiency
        batch_size = 10000
        valid = True
        for start in range(0, len(combos), batch_size):
            end = min(start + batch_size, len(combos))
            batch = combos[start:end]
            
            # Build (B, 4, 4) array
            B = end - start
            mat = np.empty((B, 4, 4), dtype=np.float64)
            for bi, (i, j, k, l) in enumerate(batch):
                mat[bi, 0] = [diff[i, 0], diff[i, 1], diff[i, 2], sq_diff[i]]
                mat[bi, 1] = [diff[j, 0], diff[j, 1], diff[j, 2], sq_diff[j]]
                mat[bi, 2] = [diff[k, 0], diff[k, 1], diff[k, 2], sq_diff[k]]
                mat[bi, 3] = [diff[l, 0], diff[l, 1], diff[l, 2], sq_diff[l]]
            
            dets = np.linalg.det(mat)
            if np.any(np.abs(dets) < 0.5):
                valid = False
                break
        
        return valid
    
    def greedy_pass(order, arr, time_limit=None):
        """Continue greedy construction from arr with given order."""
        selected = arr.copy() if len(arr) > 0 else np.empty((0, 3), dtype=np.int64)
        selected_set = set(map(tuple, selected)) if len(selected) > 0 else set()
        
        for p in order:
            if time_limit is not None and time.time() > time_limit:
                break
            p_idx = np.where((grid == p).all(axis=1))[0]
            if len(p_idx) == 0:
                continue
            p_idx = p_idx[0]
            if p_idx in selected_set:
                continue
            
            if can_add_fast(selected, p_idx, p, sq_norms[p_idx]):
                selected = np.vstack([selected, p]) if len(selected) > 0 else p[None, :]
                selected_set.add(tuple(p))
        
        return selected
    
    def local_improve(points, max_iterations=5000, time_limit=None):
        """Try to improve by swapping one point out and another in."""
        current = points.copy()
        current_set = set(map(tuple, current))
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            iterations += 1
            
            m = len(current)
            if m < 4:
                break
            
            # Try removing each point and adding a better one
            for i in range(m):
                if time_limit is not None and time.time() > time_limit:
                    break
                
                # Remove point i
                remaining = np.delete(current, i, axis=0)
                remaining_set = current_set - {tuple(current[i])}
                
                # Try adding each unselected point
                for j in range(N):
                    if j in remaining_set:
                        continue
                    p = grid[j]
                    p_idx = j
                    
                    if can_add_fast(remaining, p_idx, p, sq_norms[p_idx]):
                        # Check if this actually improves (it always does since same size)
                        # But we want to prefer points that allow future additions
                        # For now, just accept the first valid swap
                        new_points = np.vstack([remaining, p])
                        current = new_points
                        current_set = set(map(tuple, current))
                        improved = True
                        break
                if improved:
                    break
        
        return current
    
    np.random.seed(42)
    best = None
    best_len = 0
    
    # Phase 1: Multiple greedy restarts with different random orders
    num_restarts = 30
    for r in range(num_restarts):
        if time.time() > deadline - 60:
            break
        
        order = grid[np.random.permutation(N)]
        arr = np.empty((0, 3), dtype=np.int64)
        arr = greedy_pass(order, arr, time_limit=deadline - 30)
        
        if len(arr) > best_len:
            best_len = len(arr)
            best = arr.copy()
            print(f"Restart {r}: {best_len} points")
    
    # Phase 2: Local improvement on best solution
    if best is not None and time.time() < deadline - 30:
        improved = local_improve(best, max_iterations=2000, time_limit=deadline - 15)
        if len(improved) > best_len:
            best_len = len(improved)
            best = improved
            print(f"After local improve: {best_len} points")
    
    # Phase 3: More greedy restarts seeded from subsets of best
    if best is not None and time.time() < deadline - 15:
        # Try starting from random subsets of best and extending
        for r in range(20):
            if time.time() > deadline - 10:
                break
            
            subset_size = max(1, len(best) // 2)
            subset_indices = np.random.choice(len(best), subset_size, replace=False)
            subset = best[subset_indices]
            
            order = grid[np.random.permutation(N)]
            arr = subset.copy()
            arr = greedy_pass(order, arr, time_limit=deadline - 10)
            
            if len(arr) > best_len:
                best_len = len(arr)
                best = arr.copy()
                print(f"Subset restart {r}: {best_len} points")
    
    if best is None:
        # Fallback: should not happen
        best = grid[:1]
    
    return best.astype(np.int64)

# EVOLVE-BLOCK-END
