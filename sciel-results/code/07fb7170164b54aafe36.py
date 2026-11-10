# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    Maximize points in 8x8x8 grid with no 5 co-spherical.
    Uses greedy initialization + simulated annealing with vectorized energy evaluation.
    """
    n = 8
    deadline = time.time() + 580  # 20s safety margin
    
    # All grid points
    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    N = len(grid)  # 512
    sq = (grid ** 2).sum(axis=1)  # |p|^2 for each grid point
    
    # Precompute index lookup
    idx_map = {}
    for i, p in enumerate(grid):
        idx_map[tuple(p)] = i
    
    def energy(indices: np.ndarray) -> float:
        """
        Compute the number of co-spherical 5-tuples in the set.
        For each point p, check all C(k-1, 4) combinations of other points.
        A 5-tuple is co-spherical if |det| < 0.5 for the 4x4 matrix.
        Vectorized batch computation.
        """
        k = len(indices)
        if k < 5:
            return 0.0
        
        pts = grid[indices]
        sqs = sq[indices]
        total_violations = 0.0
        
        # For each point as reference, check combinations of 4 others
        for i in range(k):
            p = pts[i]
            pi_sq = sqs[i]
            others = np.delete(indices, i)
            if len(others) < 4:
                continue
            
            m = len(others)
            # Build matrix: rows are (q_j - p, |q_j|^2 - |p|^2)
            w = np.empty((m, 4), dtype=np.float64)
            w[:, 0:3] = grid[others] - p
            w[:, 3] = sq[others] - pi_sq
            
            # Compute all C(m,4) determinants
            if m <= 300:
                idxs = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
                if len(idxs) == 0:
                    continue
                sub = w[idxs]  # shape (n_combos, 4, 4)
                dets = np.linalg.det(sub)
                total_violations += np.sum(np.abs(dets) < 0.5)
            else:
                # Fallback for very large sets
                for combo in itertools.combinations(range(m), 4):
                    sub = w[list(combo)]
                    if abs(np.linalg.det(sub)) < 0.5:
                        total_violations += 1
        
        return float(total_violations)
    
    def can_add_to_set(arr_indices: np.ndarray, new_idx: int) -> bool:
        """Check if adding new_idx to the set keeps it valid."""
        m = len(arr_indices)
        if m < 4:
            return True
        p = grid[new_idx]
        pi_sq = sq[new_idx]
        others = arr_indices
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = grid[others] - p
        w[:, 3] = sq[others] - pi_sq
        if m <= 200:
            idxs = np.array(list(itertools.combinations(range(m), 4)), dtype=np.int64)
            if len(idxs) == 0:
                return True
            sub = w[idxs]
            dets = np.linalg.det(sub)
            if np.any(np.abs(dets) < 0.5):
                return False
        else:
            for combo in itertools.combinations(range(m), 4):
                sub = w[list(combo)]
                if abs(np.linalg.det(sub)) < 0.5:
                    return False
        return True
    
    def greedy_build(order: np.ndarray) -> np.ndarray:
        """Greedy: add points in given order if feasible."""
        arr = np.empty(0, dtype=np.int64)
        arr_set = set(arr.tolist())
        for idx in order:
            if idx in arr_set:
                continue
            if can_add_to_set(arr, idx):
                arr = np.append(arr, idx)
                arr_set.add(idx)
        return arr
    
    def local_search(arr: np.ndarray, max_swaps: int = 100) -> np.ndarray:
        """Try swap moves to improve the set."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        improved = True
        swaps_done = 0
        while improved and swaps_done < max_swaps and time.time() < deadline:
            improved = False
            for i in range(len(current)):
                if time.time() > deadline:
                    break
                removed = current[i]
                without = np.delete(current, i)
                for j in range(len(comp)):
                    if time.time() > deadline:
                        break
                    add_idx = comp[j]
                    if can_add_to_set(without, add_idx):
                        new_arr = np.concatenate([without, [add_idx]])
                        current = new_arr
                        current_set = set(current.tolist())
                        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                        improved = True
                        swaps_done += 1
                        break
                if improved:
                    break
        return current
    
    def local_search_add(arr: np.ndarray) -> np.ndarray:
        """Try to add more points."""
        current = arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        for idx in comp:
            if time.time() > deadline:
                break
            if can_add_to_set(current, idx):
                current = np.append(current, idx)
                current_set.add(idx)
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        return current
    
    def simulated_annealing(initial_arr: np.ndarray, max_iters: int = 5000) -> np.ndarray:
        """
        Simulated annealing to maximize set size while maintaining validity.
        Energy = number of co-spherical 5-tuples (lower is better).
        Accept invalid states with probability exp(-delta_E / T).
        """
        current = initial_arr.copy()
        current_energy = energy(current)
        best = current.copy()
        best_energy = current_energy
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            T = T_start * (cooling ** it)
            
            # Generate move: swap one point
            if len(current) == 0 or len(comp) == 0:
                break
            
            i = np.random.randint(0, len(current))
            j = np.random.randint(0, len(comp))
            
            removed = current[i]
            added = comp[j]
            
            # New set
            without = np.delete(current, i)
            new_arr = np.concatenate([without, [added]])
            
            new_energy = energy(new_arr)
            delta_E = new_energy - current_energy
            
            # Accept if better, or with probability exp(-delta_E / T)
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / T):
                current = new_arr
                current_energy = new_energy
                current_set = set(current.tolist())
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                
                # Update best if valid and larger
                if current_energy == 0 and len(current) > len(best):
                    best = current.copy()
                    best_energy = 0.0
        
        return best
    
    def simulated_annealing_grow(initial_arr: np.ndarray, max_iters: int = 3000) -> np.ndarray:
        """
        Simulated annealing that tries to grow the set.
        Moves: add a point (if valid), or swap one point.
        Energy = -len(set) + 10 * num_violations (prefer larger valid sets).
        """
        current = initial_arr.copy()
        current_set = set(current.tolist())
        all_indices = np.arange(N)
        comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
        
        best = current.copy()
        best_score = len(current)
        
        T_start = 3.0
        T_end = 0.01
        cooling = (T_end / T_start) ** (1.0 / max_iters)
        
        def score(arr):
            if len(arr) < 5:
                return -len(arr)  # No violations possible
            e = energy(arr)
            return -len(arr) + 10.0 * e
        
        current_score = score(current)
        
        for it in range(max_iters):
            if time.time() > deadline - 30:
                break
            
            T = T_start * (cooling ** it)
            
            # Choose move type: 70% swap, 30% try add
            move_type = np.random.random()
            
            if move_type < 0.7 and len(current) > 0 and len(comp) > 0:
                # Swap move
                i = np.random.randint(0, len(current))
                j = np.random.randint(0, len(comp))
                removed = current[i]
                added = comp[j]
                without = np.delete(current, i)
                new_arr = np.concatenate([without, [added]])
            elif len(comp) > 0:
                # Add move (only if it doesn't create violations, or accept with prob)
                j = np.random.randint(0, len(comp))
                added = comp[j]
                new_arr = np.concatenate([current, [added]])
            else:
                break
            
            new_score = score(new_arr)
            delta_S = new_score - current_score
            
            if delta_S < 0 or np.random.random() < np.exp(-delta_S / T):
                current = new_arr
                current_score = new_score
                current_set = set(current.tolist())
                comp = np.array([i for i in all_indices if i not in current_set], dtype=np.int64)
                
                # Update best if valid and larger
                if energy(current) == 0 and len(current) > best_score:
                    best = current.copy()
                    best_score = len(current)
        
        return best
    
    rng = np.random.RandomState(42)
    best = None
    best_len = 0
    
    # Phase 1: Multiple greedy restarts
    while time.time() < deadline - 120:
        order = rng.permutation(N)
        arr = greedy_build(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
    
    # Phase 2: Local search on best
    if best is not None and time.time() < deadline - 60:
        best = local_search(best, max_swaps=150)
        best = local_search_add(best)
        best_len = len(best)
    
    # Phase 3: Simulated annealing to escape local minima
    if best is not None and time.time() < deadline - 30:
        sa_best = simulated_annealing(best, max_iters=5000)
        if len(sa_best) > best_len:
            best = sa_best
            best_len = len(best)
    
    # Phase 4: Simulated annealing with growth
    if best is not None and time.time() < deadline - 30:
        sa_grow_best = simulated_annealing_grow(best, max_iters=3000)
        if len(sa_grow_best) > best_len:
            best = sa_grow_best
            best_len = len(best)
    
    # Phase 5: Final polish
    if best is not None and time.time() < deadline - 10:
        best = local_search(best, max_swaps=100)
        best = local_search_add(best)
        best_len = len(best)
    
    if best is None:
        best = np.array([0, 1, 2, 3], dtype=np.int64)
    
    return grid[best].astype(np.int64)


# EVOLVE-BLOCK-END
