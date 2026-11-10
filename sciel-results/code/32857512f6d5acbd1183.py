import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = 550.0  # Safety margin before 630s kill
    
    p = 7
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Element: (a, b) represents a + b*alpha
    
    # Precompute field tables
    # Represent field elements as integers 0..48: index = a + b*7
    def to_idx(a, b):
        return a + b * p
    
    def from_idx(idx):
        return idx % p, idx // p
    
    # Multiplication table
    mul_table = np.zeros((p*p, p*p), dtype=np.int32)
    for i in range(p*p):
        a1, b1 = from_idx(i)
        for j in range(p*p):
            a2, b2 = from_idx(j)
            ra = (a1*a2 + 3*b1*b2) % p
            rb = (a1*b2 + a2*b1) % p
            mul_table[i, j] = to_idx(ra, rb)
    
    # Addition table
    add_table = np.zeros((p*p, p*p), dtype=np.int32)
    for i in range(p*p):
        a1, b1 = from_idx(i)
        for j in range(p*p):
            a2, b2 = from_idx(j)
            ra = (a1 + a2) % p
            rb = (b1 + b2) % p
            add_table[i, j] = to_idx(ra, rb)
    
    # Negation table
    neg_table = np.zeros(p*p, dtype=np.int32)
    for i in range(p*p):
        a, b = from_idx(i)
        neg_table[i] = to_idx((-a) % p, (-b) % p)
    
    # Squares
    squares = set()
    for i in range(p*p):
        squares.add(int(mul_table[i, i]))
    
    # Points in F_49^2: 2401 points, indexed 0..2400
    # Point index = x_idx * 49 + y_idx where x_idx, y_idx in 0..48
    num_points = p * p * p * p  # 2401
    
    # Directions: (1, m) for m in F_49 (49 directions) + (0, 1) (1 direction) = 50 directions
    # Direction d: given by (dx, dy) where dx, dy are field element indices
    # For (1, m): dx = to_idx(1, 0), dy = m_idx
    # For (0, 1): dx = to_idx(0, 0), dy = to_idx(0, 1)
    
    dir_dx = []
    dir_dy = []
    # 49 directions: (1, m) for m = 0..48
    one_idx = to_idx(1, 0)
    zero_idx = to_idx(0, 0)
    for m in range(p*p):
        dir_dx.append(one_idx)
        dir_dy.append(m)
    # 1 direction: (0, 1)
    dir_dx.append(zero_idx)
    dir_dy.append(to_idx(0, 1))
    
    num_dirs = len(dir_dx)  # 50
    
    # For each point and each direction, precompute the line (list of other points)
    # line_points[point_idx][dir_idx] = list of 48 other points on the line
    # This is 2401 * 50 * 48 = ~5.7M entries, might be too much memory
    # Instead, compute on the fly or use a different representation
    
    # Better: for each point and direction, store the set of other points
    # Or: for each direction, for each point, compute line membership
    
    # Let's use a different approach: 
    # in_N is a boolean array of size 2401
    # For each point x and direction d, x is covered by d if all other points on line are in_N
    
    # Precompute for each point x and direction d: the list of other 48 points
    # point x = (x_idx, y_idx) where x_idx, y_idx in 0..48
    # point index = x_idx * 49 + y_idx
    
    def point_to_idx(x_i, y_i):
        return x_i * (p*p) + y_i
    
    def idx_to_point(idx):
        y_i = idx % (p*p)
        x_i = idx // (p*p)
        return x_i, y_i
    
    # Precompute line members: for each (point, direction), list of other points
    # line_members[point_idx][dir_idx] = array of 48 point indices
    line_members = np.zeros((num_points, num_dirs, p*p-1), dtype=np.int32)
    
    for x_i in range(p*p):
        for y_i in range(p*p):
            x_pt = point_to_idx(x_i, y_i)
            for d in range(num_dirs):
                dx = dir_dx[d]
                dy = dir_dy[d]
                for t in range(p*p):
                    if t == 0:
                        continue
                    # point = x + t * v
                    # x_coord = x_i + t * dx
                    # y_coord = y_i + t * dy
                    tx = int(add_table[x_i, int(mul_table[t, dx])])
                    ty = int(add_table[y_i, int(mul_table[t, dy])])
                    other_pt = point_to_idx(tx, ty)
                    line_members[x_pt, d, t-1] = other_pt
    
    # Now greedy removal
    # Start with all points in N
    in_N = np.ones(num_points, dtype=bool)
    
    # For each point, track which directions cover it
    # covered_by[point_idx][dir_idx] = True if all other points on line are in_N
    # We need at least one True for each point
    
    # Initialize: all points in N, so all lines are fully in N, so all points covered by all directions
    # covered_count[point_idx] = number of directions that cover point
    covered_count = np.full(num_points, num_dirs, dtype=np.int32)
    
    # For efficient updates: when we remove point x from N, for each direction d,
    # for each point y on the line through x in direction d (y != x), 
    # the line for y in direction d loses one point (x), so y's coverage in direction d may be broken
    
    # Let's track: for each (point, direction), how many points on the line are NOT in N
    # missing_count[point_idx][dir_idx] = number of points on line (excluding point) not in N
    # Point is covered by direction d iff missing_count[point][d] == 0
    
    missing_count = np.zeros((num_points, num_dirs), dtype=np.int32)
    
    # When we remove point x, for each direction d:
    #   for each point y on line through x in direction d (y != x):
    #     missing_count[y][d] += 1
    #     if missing_count[y][d] == 1:  # just became uncovered in direction d
    #       covered_count[y] -= 1
    
    # When we add point x back (if needed):
    #   for each direction d:
    #     for each point y on line through x in direction d (y != x):
    #       missing_count[y][d] -= 1
    #       if missing_count[y][d] == 0:  # just became covered in direction d
    #         covered_count[y] += 1
    
    # Greedy: try to remove points one by one
    # Order: try points that are "most redundant" - those with many covering directions
    
    # For efficiency, iterate multiple passes
    # In each pass, try to remove each point; if removal keeps all points covered, remove it
    
    # To check if removing point x is safe:
    #   For each direction d, for each y on line through x in direction d (y != x):
    #     if missing_count[y][d] == 0 and covered_count[y] == 1:
    #       then y would lose its last covering direction -> not safe
    #   Also, x itself: x doesn't need to be covered by being in N, but x needs to be covered
    #   x is covered if there exists d with missing_count[x][d] == 0
    #   After removing x, x's missing_count doesn't change (x is not on its own line's "other points")
    #   Wait, x IS on the line through x in direction d, but we exclude x from the check
    #   So removing x doesn't affect whether x is covered
    
    # So removing x is safe iff: for all y != x, for all d where y is on line through x in direction d:
    #   it's not the case that (missing_count[y][d] == 0 and covered_count[y] == 1)
    
    # Precompute: for each point x, for each direction d, the list of other points on the line
    # We already have line_members[x][d] = 48 points
    
    # For efficiency, let's try a simpler approach: 
    # Try removing points in a fixed order, check safety
    
    # Actually, let's try: remove all points that are "easy" to remove
    # A point x is easy to remove if for every other point y, y has at least 2 covering directions
    # (so removing x can't make any y lose its last direction)
    
    # Let's do iterative passes
    removed_count = 0
    for iteration in range(20):  # max 20 passes
        if time.time() > deadline - 30:
            break
        any_removed = False
        for x in range(num_points):
            if time.time() > deadline - 30:
                break
            if not in_N[x]:
                continue
            # Check if removing x is safe
            safe = True
            for d in range(num_dirs):
                for t_idx in range(p*p-1):
                    y = int(line_members[x, d, t_idx])
                    if missing_count[y, d] == 0 and covered_count[y] == 1:
                        safe = False
                        break
                if not safe:
                    break
            if safe:
                # Remove x
                in_N[x] = False
                removed_count += 1
                any_removed = True
                # Update missing_count and covered_count
                for d in range(num_dirs):
                    for t_idx in range(p*p-1):
                        y = int(line_members[x, d, t_idx])
                        if missing_count[y, d] == 0:
                            covered_count[y] -= 1
                        missing_count[y, d] += 1
        if not any_removed:
            break
    
    # Construct result: N = points where in_N is True
    N = []
    for idx in range(num_points):
        if in_N[idx]:
            x_i, y_i = idx_to_point(idx)
            xa, xb = from_idx(x_i)
            ya, yb = from_idx(y_i)
            N.append([[xa, xb], [ya, yb]])
    
    if len(N) == 0:
        # Fallback: return all points
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        N.append([[xa, xb], [ya, yb]])
    
    return np.array(N, dtype=np.int64)
