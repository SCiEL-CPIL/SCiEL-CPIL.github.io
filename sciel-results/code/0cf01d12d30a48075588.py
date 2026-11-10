# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points as possible from the 8x8x8 integer grid
    {0,...,7}^3 such that no five selected points are co-spherical.
    
    Uses multiple greedy strategies followed by local search to maximize
    the number of selected points.
    """
    n = 8
    deadline = time.time() + 620  # safety margin before 630s limit
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))),
                    dtype=np.int64)
    num_points = len(grid)
    
    # Precompute squared norms for all grid points
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)
    
    def can_add_fast(arr: np.ndarray, p: np.ndarray) -> bool:
        """Check if point p can be added to arr without creating 5 co-spherical points."""
        m = len(arr)
        if m < 4:
            return True
        
        # Build the 4-column matrix: (q - p, |q|^2 - |p|^2) for each q in arr
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = sq_norms[arr] - (p ** 2).sum()
        
        # Check all combinations of 4
        if m <= 4:
            return abs(np.linalg.det(w)) >= 0.5
        
        # Vectorized determinant check for all C(m,4) subsets
        combos = np.array(list(itertools.combinations(range(m), 4)))
        sub = w[combos]  # shape (C, 4, 4)
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)
    
    def greedy_random(seed: int) -> np.ndarray:
        """Greedy with random point ordering."""
        rng = np.random.RandomState(seed)
        order = grid[rng.permutation(num_points)]
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if time.time() > deadline:
                break
            if can_add_fast(arr, p):
                if len(arr) == 0:
                    arr = p[None, :]
                else:
                    arr = np.vstack([arr, p])
        return arr
    
    def greedy_farthest(seed: int) -> np.ndarray:
        """Greedy that prefers points far from the current set (spreads points)."""
        rng = np.random.RandomState(seed)
        remaining = list(range(num_points))
        rng.shuffle(remaining)
        arr = np.empty((0, 3), dtype=np.int64)
        
        for _ in range(num_points):
            if time.time() > deadline:
                break
            if not remaining:
                break
            if len(arr) == 0:
                idx = remaining.pop(0)
                arr = grid[idx][None, :]
                continue
            
            # Score each remaining point by minimum distance to current set
            best_idx = -1
            best_score = -1
            # Sample a subset if too many remaining
            check_set = remaining
            if len(remaining) > 50:
                check_set = rng.choice(remaining, size=50, replace=False).tolist()
            
            for idx in check_set:
                p = grid[idx]
                if not can_add_fast(arr, p):
                    continue
                dists = np.sqrt(((arr - p) ** 2).sum(axis=1))
                score = dists.min()
                if score > best_score:
                    best_score = score
                    best_idx = idx
            
            if best_idx >= 0:
                arr = np.vstack([arr, grid[best_idx][None, :]])
                remaining.remove(best_idx)
            else:
                break
        return arr
    
    def local_search(arr: np.ndarray, seed: int) -> np.ndarray:
        """Try to improve arr by swapping points or inserting new ones."""
        rng = np.random.RandomState(seed)
        current = arr.copy()
        improved = True
        max_iters = 200
        
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            if time.time() > deadline:
                break
            
            # Try to insert a random point from the grid
            for _ in range(50):
                if time.time() > deadline:
                    break
                idx = rng.randint(num_points)
                p = grid[idx]
                # Check if already in current
                if np.any(np.all(current == p, axis=1)):
                    continue
                if can_add_fast(current, p):
                    current = np.vstack([current, p])
                    improved = True
            
            # Try swaps: remove one point, add another
            if len(current) > 10:
                for _ in range(30):
                    if time.time() > deadline:
                        break
                    remove_idx = rng.randint(len(current))
                    trial = np.delete(current, remove_idx, axis=0)
                    idx = rng.randint(num_points)
                    p = grid[idx]
                    if np.any(np.all(trial == p, axis=1)):
                        continue
                    if can_add_fast(trial, p):
                        new_arr = np.vstack([trial, p])
                        # Only accept if same size or larger (we removed one, added one)
                        if len(new_arr) >= len(current):
                            current = new_arr
                            improved = True
        
        return current
    
    # Run multiple strategies
    best = None
    best_len = 0
    
    # Random greedy with many seeds
    for seed in range(30):
        if time.time() > deadline:
            break
        arr = greedy_random(seed)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
    
    # Farthest-first greedy
    for seed in range(10):
        if time.time() > deadline:
            break
        arr = greedy_farthest(seed)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
    
    # Local search on best solutions
    if best is not None and best_len > 0:
        for seed in range(10):
            if time.time() > deadline:
                break
            improved = local_search(best, seed)
            if len(improved) > best_len:
                best = improved
                best_len = len(improved)
    
    if best is None or len(best) == 0:
        # Fallback: return a small valid set
        return np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    
    return best.astype(np.int64)


# EVOLVE-BLOCK-END
