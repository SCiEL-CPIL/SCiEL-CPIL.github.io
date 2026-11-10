# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time
import sys


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    8x8x8 integer grid {0,...,7}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Algorithm:
    - Fast randomized greedy with diverse ordering heuristics for initial sets.
    - Precomputed squared norms to speed up can_add checks.
    - Aggressive local search with add and swap moves.
    - Deadline guard to stay within compute budget.
    
    Returns:
        points: np.ndarray of shape (k, 3) of integer grid coordinates in
            [0, 7], no five of which are co-spherical.
    """
    n = 8
    deadline = time.time() + 570  # Leave 30s safety margin for 600s limit
    
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    # Precompute squared norms
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)
    
    def can_add_vectorized(arr: np.ndarray, p: np.ndarray, arr_sq_norms: np.ndarray) -> bool:
        """
        Check if adding p to arr keeps the set sphere-free using vectorized det checks.
        arr: (m, 3) array of existing points
        p: (3,) candidate point
        arr_sq_norms: (m,) precomputed squared norms of arr
        """
        m = len(arr)
        if m < 4:
            return True
        
        # Build the 4-column matrix: [q - p, |q|^2 - |p|^2]
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr.astype(np.float64) - p.astype(np.float64)
        w[:, 3] = arr_sq_norms - (p.astype(np.float64) ** 2).sum()
        
        # For each 4-combination, check if det < 0.5
        for combo in itertools.combinations(range(m), 4):
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True
    
    def greedy_build(order: np.ndarray, sq_norms: np.ndarray) -> np.ndarray:
        """Build a valid set by greedily adding points in the given order."""
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        
        for idx in order:
            p = grid[idx]
            p_sq = sq_norms[idx]
            if can_add_vectorized(arr, p, arr_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
        
        return arr
    
    def is_valid(arr: np.ndarray) -> bool:
        """Check if the entire set is valid (no 5 co-spherical)."""
        m = len(arr)
        if m < 5:
            return True
        for combo in itertools.combinations(range(m), 5):
            pts = arr[list(combo)]
            M = np.empty((5, 5), dtype=np.float64)
            M[:, 0:3] = pts.astype(np.float64)
            M[:, 3] = (pts.astype(np.float64) ** 2).sum(axis=1)
            M[:, 4] = 1
            if abs(np.linalg.det(M)) < 0.5:
                return False
        return True
    
    # Phase 1: Multiple greedy strategies with different orderings
    np.random.seed(42)
    best = None
    best_size = 0
    
    # Strategy 1: Random permutations
    num_random = 40
    for restart in range(num_random):
        if time.time() > deadline - 280:
            break
        order = np.random.permutation(N)
        arr = greedy_build(order, sq_norms)
        if len(arr) > best_size:
            best = arr.copy()
            best_size = len(arr)
    
    # Strategy 2: Sort by various heuristics
    heuristics = [
        "manhattan",
        "max_coord",
        "min_coord",
        "dist_from_center",
        "sq_norm",
    ]
    
    for h in heuristics:
        if time.time() > deadline - 280:
            break
        if h == "manhattan":
            key = grid.sum(axis=1)
        elif h == "max_coord":
            key = grid.max(axis=1)
        elif h == "min_coord":
            key = grid.min(axis=1)
        elif h == "dist_from_center":
            center = np.array([3.5, 3.5, 3.5])
            key = ((grid - center) ** 2).sum(axis=1)
        elif h == "sq_norm":
            key = sq_norms
        else:
            continue
        
        order = np.argsort(key)
        arr = greedy_build(order, sq_norms)
        if len(arr) > best_size:
            best = arr.copy()
            best_size = len(arr)
        
        order_rev = order[::-1]
        arr_rev = greedy_build(order_rev, sq_norms)
        if len(arr_rev) > best_size:
            best = arr_rev.copy()
            best_size = len(arr_rev)
    
    if best is None:
        return grid[:10].astype(np.int64)
    
    current = best.copy()
    current_size = len(current)
    current_sq = (current.astype(np.float64) ** 2).sum(axis=1)
    
    # Phase 2: Greedy addition of remaining points
    def try_add_greedy(arr: np.ndarray, arr_sq: np.ndarray) -> tuple:
        """Try to add as many remaining points as possible."""
        current_set = set(map(tuple, arr))
        candidates = []
        cand_sq = []
        for i in range(N):
            if tuple(grid[i]) not in current_set:
                candidates.append(grid[i])
                cand_sq.append(sq_norms[i])
        
        if not candidates:
            return arr, arr_sq
        
        candidates = np.array(candidates, dtype=np.int64)
        cand_sq = np.array(cand_sq, dtype=np.float64)
        
        perm = np.random.permutation(len(candidates))
        candidates = candidates[perm]
        cand_sq = cand_sq[perm]
        
        new_arr = arr.copy()
        new_sq = arr_sq.copy()
        
        for i in range(len(candidates)):
            p = candidates[i]
            p_sq = cand_sq[i]
            if can_add_vectorized(new_arr, p, new_sq):
                new_arr = np.vstack([new_arr, p])
                new_sq = np.append(new_sq, p_sq)
        
        return new_arr, new_sq
    
    for _ in range(30):
        if time.time() > deadline - 180:
            break
        new_arr, new_sq = try_add_greedy(current, current_sq)
        if len(new_arr) > len(current):
            current = new_arr
            current_sq = new_sq
            current_size = len(current)
        else:
            break
    
    # Phase 3: Local search with swaps (remove one, try to add one or more)
    def local_search_swap(arr: np.ndarray, arr_sq: np.ndarray) -> tuple:
        """Try to improve by removing one point and adding another."""
        m = len(arr)
        if m < 1:
            return arr, arr_sq
        
        current_set = set(map(tuple, arr))
        best_arr = arr.copy()
        best_sq = arr_sq.copy()
        best_size = m
        
        for i in range(m):
            if time.time() > deadline - 120:
                break
            
            reduced = np.delete(arr, i, axis=0)
            reduced_sq = np.delete(arr_sq, i)
            reduced_set = set(map(tuple, reduced))
            
            # Try to find a replacement
            for j in range(N):
                if tuple(grid[j]) in reduced_set:
                    continue
                if tuple(grid[j]) == tuple(arr[i]):
                    continue
                
                p = grid[j]
                p_sq = sq_norms[j]
                if can_add_vectorized(reduced, p, reduced_sq):
                    new_arr = np.vstack([reduced, p])
                    new_sq = np.append(reduced_sq, p_sq)
                    if len(new_arr) > best_size:
                        best_arr = new_arr
                        best_sq = new_sq
                        best_size = len(new_arr)
                    break
        
        return best_arr, best_sq
    
    # Phase 3b: Aggressive local search - remove one, try to add multiple
    def local_search_aggressive(arr: np.ndarray, arr_sq: np.ndarray) -> tuple:
        """Remove one point and try to add as many new points as possible."""
        m = len(arr)
        if m < 1:
            return arr, arr_sq
        
        current_set = set(map(tuple, arr))
        best_arr = arr.copy()
        best_sq = arr_sq.copy()
        best_size = m
        
        for i in range(m):
            if time.time() > deadline - 100:
                break
            
            reduced = np.delete(arr, i, axis=0)
            reduced_sq = np.delete(arr_sq, i)
            reduced_set = set(map(tuple, reduced))
            
            # Try to add as many new points as possible to reduced
            temp_arr = reduced.copy()
            temp_sq = reduced_sq.copy()
            temp_set = reduced_set.copy()
            
            for j in range(N):
                if tuple(grid[j]) in temp_set:
                    continue
                if tuple(grid[j]) == tuple(arr[i]):
                    continue
                
                p = grid[j]
                p_sq = sq_norms[j]
                if can_add_vectorized(temp_arr, p, temp_sq):
                    temp_arr = np.vstack([temp_arr, p])
                    temp_sq = np.append(temp_sq, p_sq)
                    temp_set.add(tuple(grid[j]))
            
            if len(temp_arr) > best_size:
                best_arr = temp_arr
                best_sq = temp_sq
                best_size = len(temp_arr)
        
        return best_arr, best_sq
    
    for _ in range(15):
        if time.time() > deadline - 80:
            break
        new_arr, new_sq = local_search_swap(current, current_sq)
        if len(new_arr) > len(current):
            current = new_arr
            current_sq = new_sq
            current_size = len(current)
        else:
            break
    
    for _ in range(10):
        if time.time() > deadline - 60:
            break
        new_arr, new_sq = local_search_aggressive(current, current_sq)
        if len(new_arr) > len(current):
            current = new_arr
            current_sq = new_sq
            current_size = len(current)
        else:
            break
    
    # Phase 4: Final verification and cleanup
    if not is_valid(current):
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for p in current:
            p_sq = (p.astype(np.float64) ** 2).sum()
            if can_add_vectorized(arr, p, arr_sq):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
                arr_sq = np.append(arr_sq, p_sq)
        current = arr
    
    if len(current) < 1:
        current = grid[:5].astype(np.int64)
        if not is_valid(current):
            current = grid[:4].astype(np.int64)
    
    return current.astype(np.int64)


# EVOLVE-BLOCK-END
