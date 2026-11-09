# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7. A set N is a Nikodym
    set if, for every point x in F_q^2, there exists a direction v != 0 such that
    the punctured line {x + t*v : t in F_q, t != 0} is fully contained in N. The
    goal is to make N as SMALL as possible, equivalently to make its complement
    (the excluded set S) as LARGE as possible.

    Field model: F_q is built as F_p[alpha] with alpha^2 = k, where k is the
    smallest quadratic non-residue mod p. An element of F_q is the pair (a, b)
    standing for a + b*alpha, with a, b in {0, ..., p-1}.

    Strategy:
    1. Start with the baseline construction (unital-based blocking set).
    2. Attempt to grow the excluded set S via greedy addition of points that
       maintain the "weak blocking" property: for every x in S and every direction v,
       there exists t != 0 with x + t*v in S.
    
    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600  # 600 seconds
    
    # Smallest quadratic non-residue mod p (the alpha^2 = k constant).
    k_const = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k_const = c
            break
    
    # Precompute field operations
    # Element (a, b) represents a + b*alpha
    # Index = a * p + b
    # Multiplication table for F_q
    def field_mul(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        # (a1 + b1*alpha)(a2 + b2*alpha) = a1*a2 + (a1*b2 + a2*b1)*alpha + b1*b2*alpha^2
        # = (a1*a2 + b1*b2*k_const) + (a1*b2 + a2*b1)*alpha
        a_res = (a1 * a2 + b1 * b2 * k_const) % p
        b_res = (a1 * b2 + a2 * b1) % p
        return a_res * p + b_res
    
    def field_add(i, j):
        a1, b1 = divmod(i, p)
        a2, b2 = divmod(j, p)
        a_res = (a1 + a2) % p
        b_res = (b1 + b2) % p
        return a_res * p + b_res
    
    # Directions: v = (1, m) for m in F_q (q directions), and v = (0, 1) (1 direction)
    # Total q + 1 = 50 directions
    # Represent direction as (dx_idx, dy_idx) where dx, dy are field elements
    # For v = (1, m): dx = 1 (index 1), dy = m (index m)
    # For v = (0, 1): dx = 0 (index 0), dy = 1 (index 1)
    
    directions = []
    for m in range(q):
        directions.append((1, m))  # v = (1, m)
    directions.append((0, 1))      # v = (0, 1)
    
    # Precompute line membership
    # For each point x (index 0..q^2-1) and each direction v, 
    # compute the set of other points on the line through x in direction v
    # Line: {x + t*v : t in F_q}
    # x + t*v = (x_a + t*dx, x_b + t*dy)
    
    # Let's represent points as 2D indices: point_idx = x_idx * q + y_idx
    # where x_idx, y_idx in 0..q-1
    
    # For a point (x_idx, y_idx) and direction (dx_idx, dy_idx):
    # The line contains points (x_idx + t*dx_idx, y_idx + t*dy_idx) for t in F_q
    # Using field addition
    
    # Precompute for each point and direction, the list of other points on the line
    # This is a large precomputation: 2401 points * 50 directions * 49 other points = ~5.9M entries
    # Might be too slow. Let's compute on the fly or use a smarter approach.
    
    # Alternative: For a given set S (bitmask or boolean array), check if adding a point x is valid.
    # x can be added if for every direction v, there exists t != 0 such that x + t*v is in S.
    # This requires checking 50 directions * 48 possible t values = 2400 checks per point.
    # With 2401 points, that's ~5.7M checks total if we try all points.
    
    # Let's use a boolean array for S
    S = np.zeros(total_points, dtype=bool)
    
    # Initialize with the baseline construction
    # Baseline: S = points in unital U where y is NOT in y_special
    # Unital U: Trace(x) + Norm(y) = 0
    # Trace((a,b)) = 2a
    # Norm((a,b)) = a^2 - b^2*k_const
    # So: 2*x_a + y_a^2 - y_b^2*k_const = 0 (mod p)
    
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    for x_idx in range(q):
        x_a, x_b = divmod(x_idx, p)
        for y_idx in range(q):
            y_a, y_b = divmod(y_idx, p)
            in_unital = (2 * x_a + y_a * y_a - y_b * y_b * k_const) % p == 0
            in_y_special = (y_a, y_b) in y_special
            if in_unital and not in_y_special:
                point_idx = x_idx * q + y_idx
                S[point_idx] = True
    
    # Now try to grow S greedily
    # For each point not in S, check if it can be added
    # A point x can be added if for every direction v, there exists t != 0 with x + t*v in S
    
    # Precompute line neighbors for efficiency
    # For each point x and direction v, precompute the list of other points on the line
    # line_neighbors[x_idx][v_idx] = list of point indices
    
    # This is memory intensive but might be worth it
    # 2401 * 50 * 48 integers ~ 5.7M * 8 bytes = ~46 MB. Acceptable.
    
    line_neighbors = np.zeros((total_points, len(directions), q - 1), dtype=np.int32)
    
    for x_idx in range(q):
        for y_idx in range(q):
            point_idx = x_idx * q + y_idx
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            
            for v_idx, (dx_idx, dy_idx) in enumerate(directions):
                dx_a, dx_b = divmod(dx_idx, p)
                dy_a, dy_b = divmod(dy_idx, p)
                
                count = 0
                for t_idx in range(q):
                    if t_idx == 0:
                        continue
                    t_a, t_b = divmod(t_idx, p)
                    
                    # x + t*v = (x_a + t*dx, x_b + t*dy)
                    # t*dx = (t_a*dx_a + t_b*dx_b*k_const, t_a*dx_b + t_b*dx_a)
                    tx_a = (t_a * dx_a + t_b * dx_b * k_const) % p
                    tx_b = (t_a * dx_b + t_b * dx_a) % p
                    
                    ty_a = (t_a * dy_a + t_b * dy_b * k_const) % p
                    ty_b = (t_a * dy_b + t_b * dy_a) % p
                    
                    new_x_a = (x_a + tx_a) % p
                    new_x_b = (x_b + tx_b) % p
                    new_y_a = (y_a + ty_a) % p
                    new_y_b = (y_b + ty_b) % p
                    
                    new_x_idx = new_x_a * p + new_x_b
                    new_y_idx = new_y_a * p + new_y_b
                    new_point_idx = new_x_idx * q + new_y_idx
                    
                    line_neighbors[point_idx, v_idx, count] = new_point_idx
                    count += 1
    
    # Greedy growth
    # Try to add points in some order
    # We'll iterate multiple passes
    changed = True
    passes = 0
    max_passes = 100
    
    while changed and passes < max_passes:
        if time.time() > deadline - 60:  # Safety margin
            break
        changed = False
        passes += 1
        
        for point_idx in range(total_points):
            if S[point_idx]:
                continue
            
            # Check if point_idx can be added
            can_add = True
            for v_idx in range(len(directions)):
                # Check if there exists t != 0 with x + t*v in S
                found = False
                for t_count in range(q - 1):
                    neighbor_idx = line_neighbors[point_idx, v_idx, t_count]
                    if S[neighbor_idx]:
                        found = True
                        break
                if not found:
                    can_add = False
                    break
            
            if can_add:
                S[point_idx] = True
                changed = True
    
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
        # Fallback: return all points (trivially valid)
        for point_idx in range(total_points):
            x_idx = point_idx // q
            y_idx = point_idx % q
            x_a, x_b = divmod(x_idx, p)
            y_a, y_b = divmod(y_idx, p)
            points.append([[x_a, x_b], [y_a, y_b]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
