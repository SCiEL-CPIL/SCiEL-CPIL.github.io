import numpy as np
import time
import random

N = 64
TOTAL_POINTS = N * N

# Precompute all points and distance matrix
all_points = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int32)
all_pts64 = all_points.astype(np.int64)

# Distance matrix: dist_matrix[i, j] is squared distance between point i and point j
diff = all_pts64[:, np.newaxis, :] - all_pts64[np.newaxis, :, :]
dist_matrix = np.sum(diff ** 2, axis=2).astype(np.int32)
del diff

# Precompute for each point, the set of distances to all other points
# This helps in quickly checking isosceles conditions
# dist_sets[i] = sorted array of distances from point i to all others
# Actually, we can work directly with dist_matrix

def get_valid_candidates(p_idx_arr, curr_indices, dist_mat):
    """
    Vectorized check for isosceles triangles.
    p_idx_arr: array of candidate indices (M,)
    curr_indices: list of indices in current set (K,)
    dist_mat: full distance matrix (N*N, N*N)
    Returns: boolean mask of same length as p_idx_arr, True if valid.
    """
    M = len(p_idx_arr)
    if M == 0:
        return np.array([], dtype=bool)
    
    K = len(curr_indices)
    if K == 0:
        return np.ones(M, dtype=bool)
    
    curr_arr = np.array(curr_indices, dtype=np.int32)
    
    # D[i, j] = dist(p_idx_arr[i], curr_arr[j])
    # Shape: (M, K)
    D = dist_mat[p_idx_arr[:, np.newaxis], curr_arr[np.newaxis, :]]
    
    # Condition 1: No two distances from p to any two points in S can be equal.
    D_sorted = np.sort(D, axis=1)
    valid_p = np.all(np.diff(D_sorted, axis=1) != 0, axis=1)
    
    if not np.any(valid_p):
        return np.zeros(M, dtype=bool)
    
    valid_mask = valid_p.copy()
    
    # Condition 2: For any point q in S, d(p, q) must not equal d(q, r) for any r in S, r != q
    for j in range(K):
        others = np.delete(curr_arr, j)
        if len(others) == 0:
            continue
        
        V_j = dist_mat[curr_arr[j], others]
        is_in = np.isin(D[:, j], V_j)
        valid_mask &= ~is_in
        
    return valid_mask

def try_add_points(current_indices, remaining, dist_mat, max_adds=5, deadline=None):
    """Try to greedily add more points to the current set."""
    curr = list(current_indices)
    rem = remaining.copy()
    
    for _ in range(max_adds):
        if len(rem) == 0:
            break
        if deadline is not None and time.time() > deadline:
            break
        vm = get_valid_candidates(rem, curr, dist_mat)
        if not np.any(vm):
            break
        chosen = np.random.choice(rem[vm])
        curr.append(int(chosen))
        rem = np.delete(rem, np.where(rem == chosen)[0])
        
    return curr, rem

def local_search(indices, dist_mat, deadline):
    """
    Local search with multiple phases:
    1. Greedy add
    2. 1-for-2 swaps (remove 1, add 2)
    3. 1-for-1 swaps with neutral acceptance to escape local minima
    4. Final greedy add
    """
    current = list(indices)
    remaining = np.setdiff1d(np.arange(len(dist_mat)), np.array(current))
    
    # Phase 1: Greedy add
    while len(remaining) > 0 and time.time() < deadline:
        valid_mask = get_valid_candidates(remaining, current, dist_mat)
        if not np.any(valid_mask):
            break
        valid_indices = remaining[valid_mask]
        chosen_idx = np.random.choice(valid_indices)
        current.append(chosen_idx)
        remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
    
    if len(current) == 0:
        return current

    # Phase 2: Swap operations - try to find 1-for-2 improvements
    improved = True
    max_swaps = 300
    swap_count = 0
    stall_count = 0
    
    while improved and swap_count < max_swaps and time.time() < deadline:
        improved = False
        swap_count += 1
        
        if len(current) < 2:
            break
            
        # Try multiple random removals
        for attempt in range(min(30, len(current))):
            if time.time() > deadline:
                break
                
            rem_idx = np.random.randint(len(current))
            removed = current[rem_idx]
            
            new_current = current[:rem_idx] + current[rem_idx+1:]
            new_remaining = np.union1d(remaining, np.array([removed]))
            
            # Try to add up to 2 points
            temp_current, temp_remaining = try_add_points(new_current, new_remaining, dist_mat, max_adds=2, deadline=deadline)
            
            if len(temp_current) > len(current):
                current = temp_current
                remaining = temp_remaining
                improved = True
                stall_count = 0
                break
            elif len(temp_current) == len(current) and len(temp_current) > 0:
                # Accept neutral moves with decreasing probability
                accept_prob = max(0.1, 0.4 - 0.001 * stall_count)
                if np.random.random() < accept_prob:
                    current = temp_current
                    remaining = temp_remaining
                    improved = True
                    stall_count += 1
                    break
        else:
            stall_count += 1
            if stall_count > 50:
                break
    
    # Phase 3: Final greedy add after swaps
    while len(remaining) > 0 and time.time() < deadline:
        valid_mask = get_valid_candidates(remaining, current, dist_mat)
        if not np.any(valid_mask):
            break
        valid_indices = remaining[valid_mask]
        chosen_idx = np.random.choice(valid_indices)
        current.append(chosen_idx)
        remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin
    
    best_set_indices = []
    best_size = 0
    
    all_indices = np.arange(TOTAL_POINTS)
    
    # Use multiple strategies for initial construction
    strategies = ['random', 'spaced', 'diagonal', 'mixed']
    
    restart_count = 0
    max_restarts = 5000
    
    while time.time() < deadline - 3 and restart_count < max_restarts:
        restart_count += 1
        
        # Choose strategy based on restart number
        strategy = strategies[restart_count % len(strategies)]
        
        current_indices = []
        remaining = all_indices.copy()
        
        if strategy == 'random':
            np.random.shuffle(remaining)
        elif strategy == 'spaced':
            # Prefer points that are far from each other initially
            # Sort by some space-filling criterion
            shuffled = np.random.permutation(TOTAL_POINTS)
            # Try to pick points that are well-distributed
            remaining = shuffled
        elif strategy == 'diagonal':
            # Start with points along diagonals
            diag_points = []
            for d in range(2*N - 1):
                for i in range(N):
                    j = d - i
                    if 0 <= j < N:
                        idx = i * N + j
                        diag_points.append(idx)
            np.random.shuffle(diag_points)
            other_points = np.setdiff1d(all_indices, diag_points)
            np.random.shuffle(other_points)
            remaining = np.concatenate([diag_points, other_points])
        elif strategy == 'mixed':
            # Mix of random and structured
            half = TOTAL_POINTS // 2
            shuffled = np.random.permutation(TOTAL_POINTS)
            remaining = shuffled
        
        # Greedy construction
        while len(remaining) > 0 and time.time() < deadline - 15:
            valid_mask = get_valid_candidates(remaining, current_indices, dist_matrix)
            
            if not np.any(valid_mask):
                break
            
            valid_indices = remaining[valid_mask]
            
            # Use a slightly smarter selection: pick the one that has the fewest
            # conflicts with remaining points (approximate)
            if len(valid_indices) > 1 and np.random.random() < 0.3:
                # For a subset, compute how many remaining points would be invalidated
                # This is expensive, so only do it occasionally
                subset_size = min(5, len(valid_indices))
                subset = np.random.choice(valid_indices, subset_size, replace=False)
                
                # For each candidate, count how many remaining points it conflicts with
                # This is approximate and expensive, so we skip for now and use random
                chosen_idx = np.random.choice(valid_indices)
            else:
                chosen_idx = np.random.choice(valid_indices)
            
            current_indices.append(chosen_idx)
            remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
        
        # Local search
        if time.time() < deadline - 10:
            current_indices = local_search(current_indices, dist_matrix, deadline - 10)
        
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_set_indices = current_indices[:]
            # Occasionally print progress for debugging (commented out for production)
            # print(f"Restart {restart_count}: size {best_size}")
        
        # If we've found a large set, reduce exploration to focus on refinement
        if best_size > 12 and time.time() < deadline - 5:
            # Do a few more targeted restarts focused on the current best
            for _ in range(3):
                if time.time() > deadline - 3:
                    break
                # Perturb the best solution and try to improve
                perturbed = list(best_set_indices)
                if len(perturbed) > 2:
                    # Remove a random point
                    rem_idx = np.random.randint(len(perturbed))
                    perturbed.pop(rem_idx)
                    # Try to add back
                    perturbed, _ = try_add_points(perturbed, np.setdiff1d(all_indices, np.array(perturbed)), dist_matrix, max_adds=3, deadline=deadline - 3)
                
                if len(perturbed) > best_size:
                    best_size = len(perturbed)
                    best_set_indices = perturbed[:]
        
        # Check time
        if time.time() > deadline - 2:
            break
    
    # Convert indices back to coordinates
    if best_set_indices:
        best_coords = all_points[best_set_indices]
        return [(int(x), int(y)) for x, y in best_coords]
    else:
        return []

if __name__ == "__main__":
    res = construct()
    print(f"Found set of size {len(res)}")
