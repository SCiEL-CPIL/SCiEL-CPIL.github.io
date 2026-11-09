# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 550  # safety margin
    
    # Field operations for F_49 = F_7[alpha] with alpha^2 = 3
    k_const = 3
    
    def field_add(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        return ((a1 + a2) % p) * p + ((b1 + b2) % p)
    
    def field_mul(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        a_res = (a1 * a2 + b1 * b2 * k_const) % p
        b_res = (a1 * b2 + a2 * b1) % p
        return a_res * p + b_res
    
    # Directions: v = (1, m) for m in F_q, and v = (0, 1)
    # Total q + 1 = 50 directions
    directions = [(1, m) for m in range(q)] + [(0, 1)]
    
    # Precompute line membership for each point and direction
    # For point x and direction v, the line is {x + t*v : t in F_q}
    # We need to check if there's another point in S on this line
    
    # Represent points as indices 0..2400 where point_idx = x_idx * q + y_idx
    
    # For efficiency, precompute for each point and direction, the list of other points on the line
    # But this is 2401 * 50 * 48 = ~5.7M entries, which is a lot
    
    # Instead, let's compute on the fly with a boolean array for S
    
    S = np.zeros(total_points, dtype=bool)
    
    # Initialize with baseline: points in unital where y_a is not a square
    squares = set((u * u) % p for u in range(p))
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            in_unital = (2 * x_a + y_a * y_a - y_b * y_b * k_const) % p == 0
            if in_unital and y_a not in squares:
                S[x_idx * q + y_idx] = True
    
    # Now try to grow S greedily
    # For each point not in S, check if it can be added
    # A point x can be added if for every direction v, there exists t != 0 with x + t*v in S
    
    # Precompute for each point and direction, the list of other points on the line
    # line_points[x_idx][v_idx] = list of point indices (excluding x itself)
    
    line_points = np.zeros((total_points, len(directions), q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            point_idx = x_idx * q + y_idx
            
            for v_idx, (dx_idx, dy_idx) in enumerate(directions):
                dx_a, dx_b = divmod(dx_idx, p)
                dy_a, dy_b = divmod(dy_idx, p)
                
                count = 0
                for t in range(q):
                    if t == 0:
                        continue
                    # x + t*v
                    # t as field element
                    t_a, t_b = divmod(t, p)
                    
                    # t * dx
                    tdx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                    tdx_b = (t_a * dx_b + t_b * dx_a) % p
                    tdx_idx = tdx_a * p + tdx_b
                    
                    # t * dy
                    tdy_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    tdy_b = (t_a * dy_b + t_b * dy_a) % p
                    tdy_idx = tdy_a * p + tdy_b
                    
                    # x + t*v
                    nx_a = (x_a + tdx_a) % p
                    nx_b = (x_b + tdx_b) % p
                    ny_a = (y_a + tdy_a) % p
                    ny_b = (y_b + tdy_b) % p
                    
                    n_point_idx = nx_a * p * q + nx_b * q + ny_a * p + ny_b
                    
                    line_points[point_idx, v_idx, count] = n_point_idx
                    count += 1
    
    # Greedy growth: try to add points that satisfy the constraint
    # For each point not in S, check if for every direction, there's a neighbor in S
    # If yes, add it to S
    
    # We need to be careful: adding a point might make it possible to add other points
    # So we iterate until no more points can be added
    
    changed = True
    while changed:
        changed = False
        for point_idx in range(total_points):
            if S[point_idx]:
                continue
            if time.time() > deadline:
                break
            
            # Check if point can be added
            can_add = True
            for v_idx in range(len(directions)):
                # Check if any point on the line (other than point_idx) is in S
                found = False
                for t_idx in range(q - 1):
                    other_idx = line_points[point_idx, v_idx, t_idx]
                    if S[other_idx]:
                        found = True
                        break
                if not found:
                    can_add = False
                    break
            
            if can_add:
                S[point_idx] = True
                changed = True
        
        if time.time() > deadline:
            break
    
    # Build the result: N = complement of S
    points = []
    for point_idx in range(total_points):
        if not S[point_idx]:
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    if not points:
        # Fallback: return at least one point
        points = [[[0, 0], [0, 0]]]
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
