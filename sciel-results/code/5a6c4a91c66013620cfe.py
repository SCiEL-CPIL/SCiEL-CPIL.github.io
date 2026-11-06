"""No-isosceles set: numpy-vectorized greedy + local search."""
import numpy as np
import random
import time

N = 64
TIME_LIMIT = 280.0

def construct():
    start = time.time()
    deadline = start + TIME_LIMIT - 5.0

    # Precompute all grid points as int32 array of shape (N*N, 2)
    all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
    total_points = N * N

    # Precompute pairwise squared distances as a flat structure for fast access
    # dist[i, j] = squared distance between point i and point j
    # Shape: (N*N, N*N) of int32
    # This is 4096*4096*4 bytes = ~64MB, acceptable
    xs = all_points[:, 0]
    ys = all_points[:, 1]
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist_matrix = (dx * dx + dy * dy).astype(np.int32)

    def is_valid_addition(candidate_idx, current_mask):
        """Check if candidate can be added to the set defined by current_mask.
        Returns True if valid."""
        c = candidate_idx
        # Get indices of currently selected points
        current_indices = np.where(current_mask)[0]
        k = len(current_indices)
        if k < 2:
            return True

        # Distances from candidate to all current points
        d_to_curr = dist_matrix[c, current_indices]

        # Check 1: No two current points at same distance from candidate
        if np.unique(d_to_curr).size != len(d_to_curr):
            return False

        # Check 2: For each pair (i, j) in current, check if d(c,i) == d(i,j) or d(c,j) == d(i,j)
        # This is O(k^2) but we can vectorize partially
        # dist_matrix between current points: shape (k, k)
        d_curr = dist_matrix[np.ix_(current_indices, current_indices)]

        # d_to_curr as column: (k, 1)
        d_col = d_to_curr[:, None]
        # d_row: (1, k)
        d_row = d_to_curr[None, :]

        # Check: d_to_curr[i] == d_curr[i, j] for any i != j
        # i.e., d_col == d_curr (element-wise), but we need i != j (off-diagonal)
        # Also check d_to_curr[j] == d_curr[i, j], i.e., d_row == d_curr

        # Create a mask for i != j
        off_diag = ~np.eye(k, dtype=bool)

        # Check if any off-diagonal element of d_curr equals d_col[i] (row-wise)
        # d_col shape (k,1), d_curr shape (k,k)
        match1 = (d_curr == d_col) & off_diag
        if np.any(match1):
            return False

        # Check if any off-diagonal element of d_curr equals d_row[j] (column-wise)
        # d_row shape (1,k), d_curr shape (k,k)
        match2 = (d_curr == d_row) & off_diag
        if np.any(match2):
            return False

        return True

    def greedy_build(order, deadline):
        """Build a set greedily following the given order."""
        mask = np.zeros(total_points, dtype=bool)
        current_indices = []

        for idx in order:
            if time.time() > deadline:
                break
            # Quick check: distance to existing points
            if len(current_indices) < 2:
                mask[idx] = True
                current_indices.append(idx)
            else:
                if is_valid_addition(idx, mask):
                    mask[idx] = True
                    current_indices.append(idx)

        return mask

    best_mask = np.zeros(total_points, dtype=bool)
    best_size = 0

    # Multiple greedy passes with different orderings
    num_passes = 50
    for p in range(num_passes):
        if time.time() > deadline:
            break

        # Different orderings
        if p % 5 == 0:
            order = np.arange(total_points)
            np.random.shuffle(order)
        elif p % 5 == 1:
            # Sort by x
            order = np.argsort(all_points[:, 0] * 1000 + all_points[:, 1])
            np.random.shuffle(order)
        elif p % 5 == 2:
            # Sort by y
            order = np.argsort(all_points[:, 1] * 1000 + all_points[:, 0])
            np.random.shuffle(order)
        elif p % 5 == 3:
            # Sort by distance from center
            center = np.array([N // 2, N // 2])
            dists = ((all_points - center) ** 2).sum(axis=1)
            order = np.argsort(dists + np.random.rand(total_points) * 0.5)
        else:
            # Random
            order = np.arange(total_points)
            np.random.shuffle(order)

        mask = greedy_build(order, deadline)
        size = mask.sum()
        if size > best_size:
            best_size = size
            best_mask = mask.copy()

    # Local search: try to add points that weren't in the best set
    if time.time() < deadline:
        # Try adding points not in best_mask
        candidate_indices = np.where(~best_mask)[0]
        np.random.shuffle(candidate_indices)

        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in candidate_indices:
                if time.time() > deadline:
                    break
                if is_valid_addition(idx, best_mask):
                    best_mask[idx] = True
                    best_size += 1
                    improved = True
                    # Remove from candidates
                    candidate_indices = candidate_indices[candidate_indices != idx]
                    break  # Restart the loop with updated mask

        # If no more single additions, try swaps
        if time.time() < deadline:
            swap_improved = True
            while swap_improved and time.time() < deadline:
                swap_improved = False
                selected = np.where(best_mask)[0]
                unselected = np.where(~best_mask)[0]
                if len(unselected) == 0 or len(selected) < 2:
                    break

                # Try random swaps
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    if len(unselected) == 0:
                        break
                    s_idx = random.choice(selected)
                    u_idx = random.choice(unselected)

                    # Temporarily remove s_idx
                    best_mask[s_idx] = False

                    # Check if u_idx can be added
                    if is_valid_addition(u_idx, best_mask):
                        best_mask[u_idx] = True
                        selected = np.where(best_mask)[0]
                        unselected = np.where(~best_mask)[0]
                        swap_improved = True
                    else:
                        # Restore
                        best_mask[s_idx] = True

    # Convert to list of tuples
    selected_indices = np.where(best_mask)[0]
    result = [(int(all_points[i, 0]), int(all_points[i, 1])) for i in selected_indices]
    return result
