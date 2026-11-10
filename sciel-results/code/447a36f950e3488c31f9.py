# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using vectorized greedy growth with randomized restarts.
    
    Strategy:
    1. Precompute line neighbor arrays using vectorized field operations.
    2. Initialize S (complement) with a blocking-set heuristic.
    3. Run multiple randomized greedy passes: in each pass, shuffle point order,
       and for each point not in S, check via vectorized boolean ops whether it
       can be added (i.e., for every direction, at least one neighbor is in S).
    4. Keep the best S found across restarts.
    
    The key speedup: checking if a point can be added is a vectorized
    np.any(S[neighbors], axis=1).all() instead of a Python triple loop.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    n_dirs = q + 1  # 50 directions
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 570  # Safety margin before 630s kill
    
    # Smallest quadratic non-residue mod 7
    # Quadratic residues mod 7: 0,1,2,4. Non-residues: 3,5,6. Smallest: 3.
    k_const = 3
    
    # Directions: v = (1, m) for m in F_q (49 dirs), and v = (0, 1) (1 dir)
    # Represent each direction as (dx_idx, dy_idx) where idx = a*p + b
    # dx_idx, dy_idx in 0..48
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))      # v = (0, 1)
    
    # Precompute field multiplication table for t * dx and t * dy
    # t_idx in 0..48, dx_idx in 0..48
    # field_mul(t_idx, dx_idx) -> result_idx
    # We need this for computing x + t*v
    
    # Build multiplication table: mul_table[i][j] = i * j in F_q (as index)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        a1, b1 = divmod(i, p)
        for j in range(q):
            a2, b2 = divmod(j, p)
            a_res = (a1 * a2 + b1 * b2 * k_const) % p
            b_res = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = a_res * p + b_res
    
    # Precompute addition table: add_table[i][j] = i + j in F_q
    add_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        a1, b1 = divmod(i, p)
        for j in range(q):
            a2, b2 = divmod(j, p)
            a_res = (a1 + a2) % p
            b_res = (b1 + b2) % p
            add_table[i, j] = a_res * p + b_res
    
    # Precompute line neighbors: neighbors[point_idx, v_idx, t_count] = neighbor point index
    # t_count in 0..47 (48 other points on each line)
    neighbors = np.zeros((total_points, n_dirs, q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            
            for v_idx in range(n_dirs):
                dx_idx, dy_idx = directions[v_idx]
                
                count = 0
                for t_idx in range(q):
                    if t_idx == 0:
                        continue
                    # t * dx
                    tx_idx = mul_table[t_idx, dx_idx]
                    # t * dy
                    ty_idx = mul_table[t_idx, dy_idx]
                    # x + t*dx
                    new_x_idx = add_table[x_idx, tx_idx]
                    # y + t*dy
                    new_y_idx = add_table[y_idx, ty_idx]
                    new_point_idx = new_x_idx * q + new_y_idx
                    neighbors[point_idx, v_idx, count] = new_point_idx
                    count += 1
    
    # Initialize S with a heuristic blocking set
    # Use the unital-based construction as seed
    S = np.zeros(total_points, dtype=bool)
    
    # Unital: Trace(x) + Norm(y) = 0
    # Trace((a,b)) = 2a mod 7
    # Norm((a,b)) = a^2 - b^2*k_const mod 7
    # Condition: 2*x_a + y_a^2 - y_b^2*3 = 0 mod 7
    
    # Also exclude points where y is a "special" value to create a valid blocking set
    # For now, just use the unital condition as a starting point
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            if (2 * x_a + y_a * y_a - y_b * y_b * k_const) % p == 0:
                point_idx = x_idx * q + y_idx
                S[point_idx] = True
    
    best_S = S.copy()
    best_complement = total_points - np.sum(S)
    
    # Vectorized greedy growth
    # For a point x not in S, it can be added iff:
    #   for all v_idx: exists t_count such that S[neighbors[x, v_idx, t_count]] is True
    # Vectorized: 
    #   S_neighbors = S[neighbors[x]]  -> shape (n_dirs, q-1)
    #   can_add = np.all(np.any(S_neighbors, axis=1))
    
    # Even more vectorized: check ALL points at once
    # S_neighbors_all = S[neighbors]  -> shape (total_points, n_dirs, q-1)
    # This is 2401 * 50 * 48 = 5.76M bools ~ 5.76 MB. Fine.
    
    def can_add_all(S_arr):
        """Vectorized check: returns boolean array of which points can be added."""
        # S_arr: shape (total_points,)
        # neighbors: shape (total_points, n_dirs, q-1)
        # For each point, for each direction, check if any neighbor is in S
        # S[neighbors] -> shape (total_points, n_dirs, q-1)
        # any along last axis -> (total_points, n_dirs)
        # all along direction axis -> (total_points,)
        S_neighbors = S_arr[neighbors]  # (total_points, n_dirs, q-1)
        any_in_dir = np.any(S_neighbors, axis=2)  # (total_points, n_dirs)
        can_add = np.all(any_in_dir, axis=1)  # (total_points,)
        return can_add
    
    # Run multiple randomized passes
    rng = np.random.default_rng(42)
    pass_count = 0
    max_passes = 500
    
    while pass_count < max_passes:
        if time.time() > deadline:
            break
        pass_count += 1
        
        # Get order of points to try (randomized for diversity)
        order = rng.permutation(total_points)
        
        # Find points not in S
        not_in_S = np.where(~S)[0]
        if len(not_in_S) == 0:
            break
        
        # For efficiency, check all at once, then add in random order
        # But adding changes S, so we need to be careful.
        # Strategy: check which points can be added given current S,
        # then add them all at once (or in random order for diversity).
        
        # Check all points simultaneously
        can_add = can_add_all(S)
        
        # Points that are not in S and can be added
        addable = can_add & ~S
        
        if not np.any(addable):
            # No more points can be added; try a different randomization
            # Reset S to best and try again with different seed
            if np.sum(S) < np.sum(best_S):
                break
            # Perturb: remove a random subset of S to allow different growth
            if time.time() > deadline - 30:
                break
            n_remove = max(1, len(np.where(S)[0]) // 10)
            S_indices = np.where(S)[0]
            to_remove = rng.choice(S_indices, size=min(n_remove, len(S_indices)), replace=False)
            S[to_remove] = False
            continue
        
        # Add all addable points at once (greedy batch)
        S[addable] = True
        
        # Update best
        complement = total_points - np.sum(S)
        if complement > best_complement:
            best_complement = complement
            best_S = S.copy()
    
    # Final: use best_S
    S = best_S
    
    # Build the result: N = complement of S
    points_list = []
    for point_idx in range(total_points):
        if not S[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points_list.append([[x_a, x_b], [y_a, y_b]])
    
    if not points_list:
        # Fallback: return all points
        for point_idx in range(total_points):
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points_list.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points_list, dtype=np.int64)


# EVOLVE-BLOCK-END
