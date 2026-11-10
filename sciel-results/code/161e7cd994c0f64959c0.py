# EVOLVE-BLOCK-START
import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = 49
    alpha_sq = 3
    n_points = q * q

    # Precompute field operations
    def elem_add(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)

    def elem_mul(e1, e2):
        a1, b1 = divmod(e1, p)
        a2, b2 = divmod(e2, p)
        ra = (a1 * a2 + b1 * b2 * alpha_sq) % p
        rb = (a1 * b2 + a2 * b1) % p
        return ra * p + rb

    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)
    for i in range(q):
        for j in range(q):
            add_table[i, j] = elem_add(i, j)
            mul_table[i, j] = elem_mul(i, j)

    # Directions: 0..48 slopes, 49 vertical
    # Precompute line_other[idx, d, t]
    # idx = x*q + y
    # d in 0..48: slope d, vertical is d=49
    # t in 1..48 (indices 0..47)
    line_other = np.zeros((n_points, 50, 48), dtype=np.int32)
    
    xs = np.array([i // q for i in range(n_points)], dtype=np.int32)
    ys = np.array([i % q for i in range(n_points)], dtype=np.int32)
    
    for x in range(q):
        for y in range(q):
            idx = x * q + y
            a_x, b_x = divmod(x, p)
            a_y, b_y = divmod(y, p)
            
            for d in range(49):
                for t_idx, t in enumerate(range(1, q)):
                    a_t, b_t = divmod(t, p)
                    ox_a = (a_x + a_t) % p
                    ox_b = (b_x + b_t) % p
                    ox = ox_a * p + ox_b
                    
                    td = mul_table[t, d]
                    a_td, b_td = divmod(td, p)
                    oy_a = (a_y + a_td) % p
                    oy_b = (b_y + b_td) % p
                    oy = oy_a * p + oy_b
                    
                    line_other[idx, d, t_idx] = ox * q + oy
            
            # Vertical line: x + t, y
            for t_idx, t in enumerate(range(1, q)):
                a_t, b_t = divmod(t, p)
                oy_a = (a_y + a_t) % p
                oy_b = (b_y + b_t) % p
                oy = oy_a * p + oy_b
                line_other[idx, 49, t_idx] = x * q + oy

    def compute_clean_counts(S):
        # S is boolean array of shape (n_points,)
        # others_in_S[x, d, t] = S[line_other[x, d, t]]
        others_in_S = S[line_other]  # (2401, 50, 48)
        # A line is clean if no point in it is in S
        clean_lines = ~np.any(others_in_S, axis=2)  # (2401, 50)
        return np.sum(clean_lines, axis=1)

    def is_valid(S):
        return np.all(compute_clean_counts(S) > 0)

    # Local Search Strategy
    # 1. Initialize S with random points
    # 2. Repair: Remove points to satisfy constraints
    # 3. Expand: Add points if valid
    
    deadline = time.time() + 590
    best_S = None
    best_size = 0
    
    # Try multiple random initializations
    np.random.seed(42)
    
    for attempt in range(50):
        if time.time() > deadline - 10:
            break
            
        # Start with a random set of size ~1000
        init_size = 1000
        indices = np.random.choice(n_points, init_size, replace=False)
        S = np.zeros(n_points, dtype=bool)
        S[indices] = True
        
        # Repair Phase
        max_repair_iter = 1000
        for r_iter in range(max_repair_iter):
            if time.time() > deadline - 5:
                break
            
            clean_count = compute_clean_counts(S)
            unsatisfied = np.where(clean_count == 0)[0]
            
            if len(unsatisfied) == 0:
                break
            
            # Pick a random unsatisfied point
            x = unsatisfied[np.random.randint(len(unsatisfied))]
            
            # Find a point in S to remove
            # We need to remove a point s in S that is on a line through x
            # and is blocking a clean line.
            # Actually, any point s in S on a line through x might help.
            # Let's pick a random line through x that has S points, and remove one.
            
            # Find lines through x that contain S points
            # line_other[x] is (50, 48)
            # S[line_other[x]] is (50, 48) boolean
            s_on_lines = S[line_other[x]]
            # Find lines with at least one S point
            lines_with_s = np.any(s_on_lines, axis=1)
            
            if not np.any(lines_with_s):
                # This shouldn't happen if x is unsatisfied?
                # If x is unsatisfied, all lines through x have S points.
                # So there must be S points on lines through x.
                continue
            
            # Pick a random line with S points
            valid_lines = np.where(lines_with_s)[0]
            d = valid_lines[np.random.randint(len(valid_lines))]
            
            # Pick a random S point on that line
            points_on_line = line_other[x, d, :]
            s_indices_on_line = np.where(S[points_on_line])[0]
            
            if len(s_indices_on_line) == 0:
                continue
                
            s_idx_pos = s_indices_on_line[np.random.randint(len(s_indices_on_line))]
            s_idx = points_on_line[s_idx_pos]
            
            S[s_idx] = False
            
        # Check if valid
        if is_valid(S):
            current_size = np.sum(S)
            if current_size > best_size:
                best_size = current_size
                best_S = S.copy()
                
            # Expansion Phase
            # Try to add points
            # We can try to add points that are "far" from existing S points?
            # Or just random.
            max_expand_iter = 500
            for e_iter in range(max_expand_iter):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random point not in S
                not_in_S = np.where(~S)[0]
                if len(not_in_S) == 0:
                    break
                
                p_idx = not_in_S[np.random.randint(len(not_in_S))]
                
                # Tentatively add
                S[p_idx] = True
                
                if is_valid(S):
                    current_size = np.sum(S)
                    if current_size > best_size:
                        best_size = current_size
                        best_S = S.copy()
                else:
                    S[p_idx] = False

    # If no valid set found, return empty set (valid, score 0)
    # But we should find one.
    if best_S is None:
        S = np.zeros(n_points, dtype=bool)
        # S=empty is valid
        best_S = S
        best_size = 0

    # Convert to output format
    N = ~best_S
    points = []
    for x in range(q):
        for y in range(q):
            if N[x * q + y]:
                xa, xb = divmod(x, p)
                ya, yb = divmod(y, p)
                points.append([[xa, xb], [ya, yb]])

    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
