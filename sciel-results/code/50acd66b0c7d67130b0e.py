# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    n = 8
    deadline = time.time() + 600.0  # 600s limit, keep safety margin
    safety_margin = 10.0  # seconds to leave before deadline

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)
    num_points = len(grid)
    sq_norms = (grid ** 2).sum(axis=1).astype(np.float64)

    def can_add(arr, p_idx):
        """Check if grid[p_idx] can be added to arr (array of point indices)."""
        m = len(arr)
        if m < 4:
            return True
        p = grid[p_idx].astype(np.float64)
        p_sq = sq_norms[p_idx]
        
        # Compute lifted vectors for all points in arr relative to p
        arr_pts = grid[arr].astype(np.float64)
        diffs = arr_pts - p  # (m, 3)
        sq_diffs = sq_norms[arr] - p_sq  # (m,)
        
        # Check all 4-combinations
        # For speed, we can use a vectorized approach for small m
        if m <= 4:
            w = np.column_stack([diffs, sq_diffs])
            return abs(np.linalg.det(w)) >= 0.5
        
        # For larger m, check combinations
        # Use itertools.combinations
        for combo in itertools.combinations(range(m), 4):
            w = np.column_stack([diffs[list(combo)], sq_diffs[list(combo)]])
            if abs(np.linalg.det(w)) < 0.5:
                return False
        return True

    def can_add_fast(arr_indices, p_idx):
        """Faster version using precomputed data."""
        m = len(arr_indices)
        if m < 4:
            return True
        p = grid[p_idx].astype(np.float64)
        p_sq = sq_norms[p_idx]
        
        arr_pts = grid[arr_indices].astype(np.float64)
        diffs = arr_pts - p
        sq_diffs = sq_norms[arr_indices] - p_sq
        
        # Build the m x 4 matrix
        w = np.column_stack([diffs, sq_diffs])
        
        # Check all 4-combinations
        if m <= 4:
            return abs(np.linalg.det(w)) >= 0.5
        
        # For efficiency, check combinations
        for combo in itertools.combinations(range(m), 4):
            det_val = np.linalg.det(w[list(combo)])
            if abs(det_val) < 0.5:
                return False
        return True

    def can_add_precomputed(w_matrix, p_idx):
        """Check if p_idx can be added, given precomputed w for current set."""
        m = len(w_matrix)
        if m < 4:
            return True
        p = grid[p_idx].astype(np.float64)
        p_sq = sq_norms[p_idx]
        
        # Compute new row for p relative to existing points
        # w_matrix rows are (q_i - p_old, |q_i|^2 - |p_old|^2) for i in set
        # We need: for each 4-tuple from set, check det with p
        # Equivalent: check if p is cospherical with any 4 points in set
        
        # For each point q in set, compute (q - p, |q|^2 - |p|^2)
        arr_pts = grid[np.array(w_matrix[:, :3].round().astype(np.int64))].astype(np.float64) if m > 0 else np.empty((0, 3))
        
        # Actually, let's just use the original approach but with early exit
        arr_pts = grid[np.array(w_matrix[:, :3].round().astype(np.int64))].astype(np.float64)
        diffs = arr_pts - p
        sq_diffs = (arr_pts ** 2).sum(axis=1) - p_sq
        
        w_new = np.column_stack([diffs, sq_diffs])
        
        if m <= 4:
            return abs(np.linalg.det(w_new)) >= 0.5
        
        for combo in itertools.combinations(range(m), 4):
            det_val = np.linalg.det(w_new[list(combo)])
            if abs(det_val) < 0.5:
                return False
        return True

    def is_valid_set(indices):
        """Check if a set of point indices is valid (no 5 co-spherical)."""
        m = len(indices)
        if m < 5:
            return True
        for i in range(4, m):
            p_idx = indices[i]
            earlier = indices[:i]
            if not can_add_fast(earlier, p_idx):
                return False
        return True

    def greedy_fill(order, start_set=None, time_limit=None):
        """Greedy fill: add points in given order, starting from start_set."""
        if start_set is None:
            arr = []
        else:
            arr = list(start_set)
        
        used = set(arr)
        for p_idx in order:
            if p_idx in used:
                continue
            if can_add_fast(arr, p_idx):
                arr.append(p_idx)
                used.add(p_idx)
            if time_limit is not None and time.time() > time_limit:
                break
        return arr

    def local_search(indices, num_iterations=100, time_limit=None):
        """Local search: try to improve the set by swapping points."""
        current = list(indices)
        current_set = set(current)
        
        for _ in range(num_iterations):
            if time_limit is not None and time.time() > time_limit:
                break
            improved = False
            # Try removing each point and adding a new one
            for i in range(len(current)):
                if time_limit is not None and time.time() > time_limit:
                    break
                # Remove current[i]
                without_i = current[:i] + current[i+1:]
                # Try to add points not in the set
                for p_idx in range(num_points):
                    if p_idx in current_set:
                        continue
                    if can_add_fast(without_i, p_idx):
                        new_set = without_i + [p_idx]
                        # Check if this is valid (should be by construction)
                        # Try to extend greedily
                        extended = greedy_fill(
                            [x for x in range(num_points) if x not in set(new_set)],
                            start_set=new_set,
                            time_limit=time_limit
                        )
                        if len(extended) > len(current):
                            current = extended
                            current_set = set(current)
                            improved = True
                            break
                if improved:
                    break
            if not improved:
                break
        return current

    def local_search_fast(indices, time_limit=None):
        """Faster local search: try all single swaps, keep best improvement."""
        current = list(indices)
        current_set = set(current)
        current_len = len(current)
        
        if time_limit is not None and time.time() > time_limit:
            return current
        
        best = current
        best_len = current_len
        
        # Precompute which points can replace which
        for i in range(len(current)):
            if time_limit is not None and time.time() > time_limit - 5:
                break
            without_i = current[:i] + current[i+1:]
            # Find all points that can replace current[i]
            candidates = []
            for p_idx in range(num_points):
                if p_idx in current_set:
                    continue
                if can_add_fast(without_i, p_idx):
                    candidates.append(p_idx)
            
            if not candidates:
                continue
            
            # Try each candidate, extend greedily
            for p_idx in candidates:
                if time_limit is not None and time.time() > time_limit - 5:
                    break
                new_set = without_i + [p_idx]
                new_set_set = set(new_set)
                extended = greedy_fill(
                    [x for x in range(num_points) if x not in new_set_set],
                    start_set=new_set,
                    time_limit=time_limit
                )
                if len(extended) > best_len:
                    best = extended
                    best_len = len(extended)
                    current = extended
                    current_set = set(current)
                    current_len = current_len + 1
                    break  # Restart with improved set
        
        return best

    # Generate diverse orderings
    def manhattan_order():
        center = 3.5
        dists = np.abs(grid - center).sum(axis=1)
        return np.argsort(dists, kind='stable')

    def max_coord_order():
        # Points with max coordinate spread
        scores = grid.max(axis=1)
        return np.argsort(scores, kind='stable')

    def min_coord_order():
        scores = grid.min(axis=1)
        return np.argsort(scores, kind='stable')

    def random_order(seed):
        rng = np.random.RandomState(seed)
        return rng.permutation(num_points)

    def diagonal_order():
        # Order by sum of coordinates
        sums = grid.sum(axis=1)
        return np.argsort(sums, kind='stable')

    def euclidean_order():
        center = 3.5
        dists = np.sqrt(((grid - center) ** 2).sum(axis=1))
        return np.argsort(dists, kind='stable')

    def product_order():
        scores = grid.prod(axis=1)
        return np.argsort(scores, kind='stable')

    def parity_order():
        # Order by parity of coordinate sum
        parities = grid.sum(axis=1) % 2
        return np.argsort(parities * 1000 + grid.sum(axis=1), kind='stable')

    # Try multiple strategies
    best = None
    best_len = 0

    strategies = [
        manhattan_order,
        max_coord_order,
        min_coord_order,
        diagonal_order,
        euclidean_order,
        product_order,
        parity_order,
        lambda: random_order(0),
        lambda: random_order(1),
        lambda: random_order(2),
        lambda: random_order(3),
        lambda: random_order(4),
        lambda: random_order(5),
        lambda: random_order(6),
        lambda: random_order(7),
    ]

    for strat in strategies:
        if time.time() > deadline - safety_margin:
            break
        order = strat()
        arr = greedy_fill(order)
        if len(arr) > best_len:
            best = arr
            best_len = len(arr)
        # Try local search on the best found so far
        if best is not None and len(best) > 0:
            if time.time() > deadline - safety_margin - 30:
                break
            improved = local_search_fast(best, time_limit=deadline - safety_margin)
            if len(improved) > best_len:
                best = improved
                best_len = len(improved)

    if best is None:
        # Fallback: just return a small valid set
        return grid[:10].astype(np.int64)

    return grid[np.array(best)].astype(np.int64)


# EVOLVE-BLOCK-END
