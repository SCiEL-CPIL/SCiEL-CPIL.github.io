# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Select as many points from the 8x8x8 grid {0,...,7}^3 such that no five
    are co-spherical. The test: for each point p and every 4 earlier points
    q1..q4, the 4x4 matrix with rows (q_i - p, |q_i|^2 - |p|^2) must have
    determinant with absolute value >= 0.5.

    Strategy: Optimized greedy with vectorized determinant checks,
    diverse ordering heuristics (including dispersive), and local search.
    """
    n = 8
    deadline = time.time() + 590  # leave 40s margin for safety

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq_norms = (grid.astype(np.float64) ** 2).sum(axis=1)  # precompute |q|^2 for all grid points

    def can_add_fast(arr: np.ndarray, p: np.ndarray, arr_sq_norms: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        # Build the m x 4 matrix: columns are (dx, dy, dz, |q|^2 - |p|^2)
        p_f = p.astype(np.float64)
        p_sq = (p_f ** 2).sum()
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr.astype(np.float64) - p_f
        w[:, 3] = arr_sq_norms - p_sq

        # Generate all 4-combinations
        combos = np.array(list(itertools.combinations(range(m), 4)))
        if len(combos) == 0:
            return True
        sub = w[combos]
        # Compute determinants in batch
        dets = np.linalg.det(sub)
        return not np.any(np.abs(dets) < 0.5)

    def greedy_with_order(order: np.ndarray) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        for idx in order:
            p = grid[idx]
            if can_add_fast(arr, p, arr_sq):
                if len(arr) == 0:
                    arr = p[None, :]
                    arr_sq = np.array([sq_norms[idx]])
                else:
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, sq_norms[idx])
        return arr

    def dispersive_greedy() -> np.ndarray:
        """
        Greedy selection that maximizes the minimum distance to the existing set.
        This helps avoid clustering, which reduces the risk of 5 points being
        coplanar or cospherical.
        """
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        remaining = set(range(N))
        
        # Start with a point near the center
        center = np.array([3.5, 3.5, 3.5])
        dists = np.linalg.norm(grid - center, axis=1)
        start_idx = np.argmin(dists)
        arr = grid[start_idx][None, :]
        arr_sq = np.array([sq_norms[start_idx]])
        remaining.remove(start_idx)
        
        while time.time() < deadline:
            if not remaining: break
            
            rem_indices = list(remaining)
            if not rem_indices: break
            rem_grid = grid[rem_indices]
            
            # Compute distances from each rem point to each arr point
            # (N_rem, 1, 3) - (1, k, 3) -> (N_rem, k, 3)
            dists_mat = np.linalg.norm(rem_grid[:, None, :] - arr[None, :, :], axis=2)
            min_dists = dists_mat.min(axis=1)
            
            # Try the top 5 most dispersive points
            top_k = min(5, len(rem_indices))
            top_indices = np.argsort(-min_dists)[:top_k]
            
            added = False
            for i in top_indices:
                gi = rem_indices[i]
                p = grid[gi]
                if can_add_fast(arr, p, arr_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, sq_norms[gi])
                    remaining.remove(gi)
                    added = True
                    break
            
            if not added:
                break
                
        return arr

    # Precompute various orderings
    center = np.array([3.5, 3.5, 3.5])
    dists_from_center = np.linalg.norm(grid - center, axis=1)
    manhattan = np.abs(grid - 3.5).sum(axis=1)
    sum_coords = grid.sum(axis=1)
    max_coord = grid.max(axis=1)
    min_coord = grid.min(axis=1)

    # Generate diverse orderings
    orderings = []
    # Distance-based
    orderings.append(np.argsort(dists_from_center))
    orderings.append(np.argsort(-dists_from_center))
    orderings.append(np.argsort(manhattan))
    orderings.append(np.argsort(-manhattan))
    orderings.append(np.argsort(sum_coords))
    orderings.append(np.argsort(-sum_coords))
    orderings.append(np.argsort(max_coord))
    orderings.append(np.argsort(-max_coord))
    orderings.append(np.argsort(min_coord))
    orderings.append(np.argsort(-min_coord))

    # Random orderings with different seeds
    for seed in [42, 123, 777, 999, 2024, 3141, 2718, 1618, 1414, 1732]:
        rng = np.random.RandomState(seed)
        orderings.append(rng.permutation(N))

    best = np.empty((0, 3), dtype=np.int64)

    # Phase 1: Try all precomputed orderings
    for order in orderings:
        if time.time() > deadline - 30:
            break
        arr = greedy_with_order(order)
        if len(arr) > len(best):
            best = arr

    # Phase 2: Dispersive Greedy Restarts
    # Run dispersive greedy multiple times with different starting points?
    # The current dispersive_greedy always starts with the same center point.
    # We can modify it to take a start index.
    
    def dispersive_greedy_start(start_idx: int) -> np.ndarray:
        arr = np.empty((0, 3), dtype=np.int64)
        arr_sq = np.empty(0, dtype=np.float64)
        remaining = set(range(N))
        
        arr = grid[start_idx][None, :]
        arr_sq = np.array([sq_norms[start_idx]])
        remaining.remove(start_idx)
        
        while time.time() < deadline:
            if not remaining: break
            
            rem_indices = list(remaining)
            if not rem_indices: break
            rem_grid = grid[rem_indices]
            
            # Compute distances from each rem point to each arr point
            dists_mat = np.linalg.norm(rem_grid[:, None, :] - arr[None, :, :], axis=2)
            min_dists = dists_mat.min(axis=1)
            
            # Try the top 5 most dispersive points
            top_k = min(5, len(rem_indices))
            top_indices = np.argsort(-min_dists)[:top_k]
            
            added = False
            for i in top_indices:
                gi = rem_indices[i]
                p = grid[gi]
                if can_add_fast(arr, p, arr_sq):
                    arr = np.vstack([arr, p])
                    arr_sq = np.append(arr_sq, sq_norms[gi])
                    remaining.remove(gi)
                    added = True
                    break
            
            if not added:
                break
                
        return arr

    # Try dispersive greedy with different starts
    start_candidates = [0, N-1, N//2, np.argmin(dists_from_center), np.argmax(dists_from_center)]
    for s in start_candidates:
        if time.time() > deadline - 30:
            break
        arr = dispersive_greedy_start(s)
        if len(arr) > len(best):
            best = arr

    # Phase 3: More random restarts
    rng2 = np.random.RandomState(555)
    for i in range(100):
        if time.time() > deadline - 30:
            break
        perm = rng2.permutation(N)
        arr = greedy_with_order(perm)
        if len(arr) > len(best):
            best = arr
            
        # Alternate with dispersive
        if time.time() > deadline - 30:
            break
        s = rng2.randint(N)
        arr = dispersive_greedy_start(s)
        if len(arr) > len(best):
            best = arr

    # Phase 4: Local search - try to improve by swapping
    if time.time() < deadline - 40 and len(best) > 0:
        best_set = set(map(tuple, best.tolist()))
        remaining_indices = [i for i in range(N) if tuple(grid[i]) not in best_set]
        rng3 = np.random.RandomState(888)

        improved = True
        max_iters = 50
        for it in range(max_iters):
            if time.time() > deadline - 40 or not improved:
                break
            improved = False
            k = len(best)
            # Try removing each point (sample if k is large)
            indices_to_try = list(range(k))
            if k > 30:
                indices_to_try = rng3.choice(k, size=30, replace=False)
            
            for idx in indices_to_try:
                if time.time() > deadline - 40:
                    break
                # Remove point at idx
                trial = np.delete(best, idx, axis=0)
                trial_sq = np.delete((best.astype(np.float64) ** 2).sum(axis=1), idx)
                
                # Try adding candidates from remaining
                # Sample candidates to speed up
                cand_sample = rng3.choice(len(remaining_indices), size=min(50, len(remaining_indices)), replace=False)
                for c_i in cand_sample:
                    cand_idx = remaining_indices[c_i]
                    if cand_idx in best_set:
                        continue
                    p = grid[cand_idx]
                    if can_add_fast(trial, p, trial_sq):
                        trial = np.vstack([trial, p])
                        trial_sq = np.append(trial_sq, sq_norms[cand_idx])
                        best_set.add(tuple(p))
                        if cand_idx in remaining_indices:
                            remaining_indices.remove(cand_idx)
                        improved = True
                        break
                
                if len(trial) > len(best):
                    best = trial
                    best_set = set(map(tuple, best.tolist()))
                    remaining_indices = [i for i in range(N) if tuple(grid[i]) not in best_set]
                    improved = True
                    break

    # Phase 5: Final random restarts with time remaining
    if time.time() < deadline - 20:
        rng4 = np.random.RandomState(4242)
        for i in range(200):
            if time.time() > deadline - 20:
                break
            perm = rng4.permutation(N)
            arr = greedy_with_order(perm)
            if len(arr) > len(best):
                best = arr
            
            if time.time() > deadline - 20:
                break
            s = rng4.randint(N)
            arr = dispersive_greedy_start(s)
            if len(arr) > len(best):
                best = arr

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
