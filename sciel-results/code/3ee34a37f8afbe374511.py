import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    start_time = time.time()
    deadline = start_time + 600  # safety margin
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element (a, b) represents a + b*alpha
    # Addition: (a,b) + (c,d) = (a+c mod p, b+d mod p)
    # Multiplication: (a,b)*(c,d) = (ac+bd*3, ad+bc) mod p
    
    # Precompute field multiplication table
    mul_table = np.zeros((q, q), dtype=np.int64)
    for i in range(q):
        a1, b1 = i // p, i % p
        for j in range(q):
            a2, b2 = j // p, j % p
            ra = (a1 * a2 + b1 * b2 * 3) % p
            rb = (a1 * b2 + a2 * b1) % p
            mul_table[i, j] = ra * p + rb
    
    # Directions: (1, m) for m in F_49, and (0, 1)
    # Direction (dx_a, dx_b, dy_a, dy_b)
    directions = []
    for m in range(q):
        directions.append((1, 0, m // p, m % p))  # (1, m)
    directions.append((0, 1, 0, 0))  # (0, 1)
    num_dirs = len(directions)  # 50
    
    # For each point and direction, precompute the line (set of other points)
    # Point index: idx = xa * q + xb * p + ya * p + yb... 
    # Actually, let's index points as (x, y) where x, y in [0, q)
    # x = xa * p + xb, y = ya * p + yb
    # Point index = x * q + y
    
    # For each point (x, y) and direction (dx_a, dx_b, dy_a, dy_b):
    # Line points: (x + t*dx, y + t*dy) for t in F_49, t != 0
    # We need: for point (x,y), direction d, the set of other points on the line
    
    # Precompute: for each point index and direction, list of other point indices on that line
    # This is 2401 * 50 * 48 entries... that's 5.7 million, manageable
    
    # Actually, let's be smarter. For a given direction (dx, dy) and point (x, y),
    # the line is {(x + t*dx, y + t*dy) : t in F_49}
    # The other points are t = 1, 2, ..., q-1
    
    # Let's precompute for each direction, for each point, the set of other points
    # But that's a lot of memory. Instead, let's compute on the fly or use a different approach.
    
    # Alternative approach: iterative removal
    # S starts as all points. For each point, check if it's "isolated" on some line.
    # A point x is isolated on direction d if no other point of S is on line(x, d).
    
    # To check if x is isolated on direction d: for all t != 0, point (x + t*dx, y + t*dy) not in S.
    
    # Let's represent S as a boolean array of size q*q
    in_S = np.ones(q * q, dtype=bool)
    
    def point_idx(x, y):
        return x * q + y
    
    def get_xy(idx):
        return idx // q, idx % q
    
    # For a point (x, y) and direction (dx_a, dx_b, dy_a, dy_b),
    # compute the other points on the line
    def line_other_points(x, y, dx_a, dx_b, dy_a, dy_b):
        """Return list of indices of other points on the line through (x,y) in given direction."""
        result = []
        for t in range(1, q):
            # t as field element
            t_a = t // p
            t_b = t % p
            # t * dx = (t_a*dx_a + t_b*dx_b*3, t_a*dx_b + t_b*dx_a) mod p
            tx_a = (t_a * dx_a + t_b * dx_b * 3) % p
            tx_b = (t_a * dx_b + t_b * dx_a) % p
            ty_a = (t_a * dy_a + t_b * dy_b * 3) % p
            ty_b = (t_a * dy_b + t_b * dy_a) % p
            # new point: (x + tx, y + ty)
            x_new = (x + tx_a * p + tx_b) % q  # WRONG! x is already in [0,q)
            # Actually x = xa * p + xb, and tx = tx_a * p + tx_b
            # x + tx = (xa + tx_a) * p + (xb + tx_b) mod p for each component
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            nxa = (xa + tx_a) % p
            nxb = (xb + tx_b) % p
            nya = (ya + ty_a) % p
            nyb = (yb + ty_b) % p
            nx = nxa * p + nxb
            ny = nya * p + nyb
            result.append(nx * q + ny)
        return result
    
    # This is too slow. Let me precompute line memberships.
    # For each direction d and each point x, precompute the set of other points on line(x, d).
    # Store as a list of arrays: line_sets[d][x] = array of other point indices
    
    print("Precomputing line memberships...")
    line_sets = []  # line_sets[d][x] = list of other point indices
    for d in range(num_dirs):
        dx_a, dx_b, dy_a, dy_b = directions[d]
        d_lines = []
        for x in range(q):
            for y in range(q):
                pts = []
                for t in range(1, q):
                    t_a = t // p
                    t_b = t % p
                    tx_a = (t_a * dx_a + t_b * dx_b * 3) % p
                    tx_b = (t_a * dx_b + t_b * dx_a) % p
                    ty_a = (t_a * dy_a + t_b * dy_b * 3) % p
                    ty_b = (t_a * dy_b + t_b * dy_a) % p
                    xa, xb = x // p, x % p
                    ya, yb = y // p, y % p
                    nxa = (xa + tx_a) % p
                    nxb = (xb + tx_b) % p
                    nya = (ya + ty_a) % p
                    nyb = (yb + ty_b) % p
                    nx = nxa * p + nxb
                    ny = nya * p + nyb
                    pts.append(nx * q + ny)
                d_lines.append(pts)
        line_sets.append(d_lines)
        print(f"  Direction {d+1}/{num_dirs} done")
    
    # Now iteratively remove points that are not isolated on any direction
    print("Starting iterative removal...")
    changed = True
    iteration = 0
    while changed:
        changed = False
        iteration += 1
        for idx in range(q * q):
            if not in_S[idx]:
                continue
            x, y = get_xy(idx)
            # Check if idx is isolated on any direction
            isolated = False
            for d in range(num_dirs):
                other_pts = line_sets[d][idx]
                # Check if any other point is in S
                if not any(in_S[pt] for pt in other_pts):
                    isolated = True
                    break
            if not isolated:
                in_S[idx] = False
                changed = True
        print(f"  Iteration {iteration}: |S| = {np.sum(in_S)}")
        if time.time() > deadline - 60:
            break
    
    # Build the Nikodym set N = complement of S
    points = []
    for idx in range(q * q):
        if not in_S[idx]:
            x, y = get_xy(idx)
            xa, xb = x // p, x % p
            ya, yb = y // p, y % p
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
