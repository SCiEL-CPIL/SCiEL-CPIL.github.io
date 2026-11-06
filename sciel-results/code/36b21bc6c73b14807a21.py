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

# Precompute conflict graph using bitsets
# conflict[i] is a bitset where bit j is set if points i and j form an isosceles triangle with some third point
# Two points i,j conflict if there exists k != i,j such that d(i,k) == d(j,k) or d(i,k) == d(i,j) or d(j,k) == d(i,j)
# Actually, for no-isosceles: three points a,b,c form isosceles if any two distances are equal.
# So i and j conflict if there exists k such that {d(i,k), d(j,k), d(i,j)} has a duplicate.
# That is: d(i,k) == d(j,k) OR d(i,k) == d(i,j) OR d(j,k) == d(i,j)

# For efficiency, let's precompute for each pair (i,j) whether they conflict.
# But that's 4096^2 / 2 ~ 8 million pairs, and for each we'd need to check all 4096 k's.
# That's 32 billion operations - too slow.

# Alternative: For each k, group points by distance to k. Points in the same group conflict with each other.
# Also, for each pair (i,j), if d(i,j) equals d(i,k) or d(j,k) for some k, they conflict.
# This is complex. Let's use a simpler but still effective approach.

# Actually, let me reconsider. The conflict condition for adding point p to set S is:
# For any two points a,b in S: d(p,a) != d(p,b), d(p,a) != d(a,b), d(p,b) != d(a,b)
# This is what get_valid_candidates checks.

# For branch-and-bound, I need a fast way to check if a point can be added to a partial solution.
# Let me keep the existing get_valid_candidates but optimize it, and add a B&B layer.

def construct():
    start_time = time.time()
    # Safety margin: 300s wall-clock limit, use 270s for safety
    deadline = start_time + 270
    
    best_set_indices = []
    best_size = 0
    
    # Precompute indices for all points
    all_indices = np.arange(TOTAL_POINTS)
    
    num_restarts = 2000  # We will break based on time
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # 1. Greedy Construction with Vectorized Validity Checks
        current_indices = []
        remaining = all_indices.copy()
        
        # Randomize the order of remaining points for diversity
        np.random.shuffle(remaining)
        
        while len(remaining) > 0 and time.time() < deadline - 15:
            # Check which remaining points are compatible with current set
            valid_mask = get_valid_candidates(remaining, current_indices, dist_matrix)
            
            if not np.any(valid_mask):
                break
            
            valid_indices = remaining[valid_mask]
            
            # Choose a random valid index
            chosen_idx = np.random.choice(valid_indices)
            
            current_indices.append(chosen_idx)
            
            # Remove chosen from remaining
            remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
        
        # 2. Local Search with B&B-inspired swap
        if time.time() < deadline - 5:
            current_indices = local_search(current_indices, dist_matrix, deadline - 5)
            
        if len(current_indices) > best_size:
            best_size = len(current_indices)
            best_set_indices = current_indices[:]
            
    # Convert indices back to coordinates
    if best_set_indices:
        best_coords = all_points[best_set_indices]
        return [(int(x), int(y)) for x, y in best_coords]
    else:
        return []

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
    if K > 1:
        D_sorted = np.sort(D, axis=1)
        valid_p = np.all(np.diff(D_sorted, axis=1) != 0, axis=1)
    else:
        valid_p = np.ones(M, dtype=bool)
        
    valid_mask = valid_p.copy()
    
    # Condition 2 & 3: For any point q in S (curr_arr[j]), the distance from p to q 
    # must not equal the distance from q to any other point r in S.
    for j in range(K):
        others = np.delete(curr_arr, j)
        if len(others) == 0:
            continue
        
        V_j = dist_mat[curr_arr[j], others]
        is_in = np.isin(D[:, j], V_j)
        valid_mask &= ~is_in
        
    return valid_mask

def local_search(indices, dist_mat, deadline):
    """
    Local search to improve the set size.
    Strategy: Try to remove 1 point and add 2 points (net +1).
    """
    current = list(indices)
    curr_set = set(current)
    remaining = np.setdiff1d(np.arange(len(dist_mat)), np.array(current))
    
    improved = True
    max_swaps = 200
    swap_count = 0
    
    # Phase 1: Try to add points greedily first
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

    # Phase 2: Swap operations
    while improved and swap_count < max_swaps and time.time() < deadline:
        improved = False
        swap_count += 1
        
        if len(current) < 2:
            break
            
        # Try a few random removals
        for attempt in range(min(15, len(current))):
            if time.time() > deadline:
                break
                
            rem_idx = np.random.randint(len(current))
            removed = current[rem_idx]
            
            new_current = current[:rem_idx] + current[rem_idx+1:]
            new_remaining = np.union1d(remaining, np.array([removed]))
            
            temp_current, temp_remaining = try_add_points(new_current, new_remaining, dist_mat, max_adds=2)
            
            if len(temp_current) > len(current):
                current = temp_current
                remaining = temp_remaining
                improved = True
                break
            elif len(temp_current) == len(current) and random.random() < 0.15:
                current = temp_current
                remaining = temp_remaining
                improved = True
                break

    return current

def try_add_points(current, remaining, dist_mat, max_adds):
    """
    Try to add up to max_adds points to current from remaining.
    """
    current = list(current)
    remaining = np.array(remaining)
    
    for _ in range(max_adds):
        if len(remaining) == 0:
            break
        valid_mask = get_valid_candidates(remaining, current, dist_mat)
        if not np.any(valid_mask):
            break
        valid_indices = remaining[valid_mask]
        chosen_idx = np.random.choice(valid_indices)
        current.append(chosen_idx)
        remaining = np.delete(remaining, np.where(remaining == chosen_idx)[0])
        
    return current, remaining
