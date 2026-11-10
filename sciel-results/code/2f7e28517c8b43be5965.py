# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a small Nikodym set in F_49^2.
    Start with the Hermitian unital construction, then use local search
    to greedily add more excluded points while maintaining validity.
    """
    start = time.time()
    deadline = start + 600  # safety margin before 630s

    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3

    # Precompute field operations as lookup tables
    # Element index = a + b*7 where elem = a + b*alpha
    add_table = np.zeros((q, q), dtype=np.int32)
    mul_table = np.zeros((q, q), dtype=np.int32)

    for a1 in range(p):
        for b1 in range(p):
            idx1 = a1 + b1 * p
            for a2 in range(p):
                for b2 in range(p):
                    idx2 = a2 + b2 * p
                    add_table[idx1, idx2] = ((a1 + a2) % p) + ((b1 + b2) % p) * p
                    ra = (a1 * a2 + b1 * b2 * k) % p
                    rb = (a1 * b2 + a2 * b1) % p
                    mul_table[idx1, idx2] = ra + rb * p

    # Precompute line structures
    # For each direction m (0..48 for v=(1,m), 49 for v=(0,1)):
    # For each point x_idx, list of other points on the line
    # Total: 50 * 2401 * 48 = ~5.76M entries

    total_points = q * q  # 2401
    num_dirs = q + 1  # 50

    # For each direction and point, store the line's other points
    # line_points[dir][pt_idx] = list of other point indices on the line
    line_points = []
    for d in range(num_dirs):
        dir_points = []
        for pt_idx in range(total_points):
            x_idx = pt_idx // q
            y_idx = pt_idx % q
            others = []
            for t_idx in range(q):
                if t_idx == 0:
                    continue
                if d < q:  # v = (1, m)
                    nx = add_table[x_idx, t_idx]
                    tm = mul_table[t_idx, d]
                    ny = add_table[y_idx, tm]
                else:  # v = (0, 1)
                    nx = x_idx
                    ny = add_table[y_idx, t_idx]
                others.append(nx * q + ny)
            dir_points.append(others)
        line_points.append(dir_points)

    # Initial construction: Hermitian unital with special Y set
    # U = {(x,y) : Trace(x) + Norm(y) = 0}
    # Y = {(u^2 mod p, v) : u,v in F_7}
    # Excluded S = {points in U where y NOT in Y}

    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    excluded = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            x_idx = xa + xb * p
            tr_x = (2 * xa) % p
            for ya in range(p):
                for yb in range(p):
                    y_idx = ya + yb * p
                    norm_y = (ya * ya - yb * yb * k) % p
                    pt_idx = x_idx * q + y_idx
                    if (tr_x + norm_y) % p == 0 and (ya, yb) not in y_special:
                        excluded[pt_idx] = True

    initial_size = excluded.sum()
    print(f"Initial excluded size: {initial_size}")

    # Helper: check if adding a point to excluded maintains validity
    # For a point p to be excluded, for every other point x on any line through p,
    # x must still have some direction where all other points on that line are not excluded.
    # More precisely: after excluding p, for each point x (including p itself),
    # x must have at least one direction d such that no excluded point is on the line
    # through x in direction d (other than x itself if x is excluded).

    def check_point_valid(x_idx, excl):
        """Check if point x_idx is valid (has a direction with no other excluded points on line)."""
        for d in range(num_dirs):
            valid = True
            for other_idx in line_points[d][x_idx]:
                if excl[other_idx]:
                    valid = False
                    break
            if valid:
                return True
        return False

    def check_all_valid(excl):
        """Check if all points are valid given excluded set."""
        for pt_idx in range(total_points):
            if not check_point_valid(pt_idx, excl):
                return False
        return True

    # Local search: try to add more excluded points
    # A point can be added to excluded if:
    # 1. It's not already excluded
    # 2. After adding it, all points remain valid
    
    # Key insight: adding point p to excluded only affects validity of points
    # that share a line with p in some direction. Specifically, for a direction d,
    # points on the same line as p in direction d might lose that direction as valid.
    
    # But we need to be careful: a point x loses direction d as valid only if
    # p is on the line through x in direction d AND x doesn't have another valid direction.
    
    # For efficiency, let's try points one by one in random order.
    
    np.random.seed(42)
    order = np.random.permutation(total_points)
    
    added_count = 0
    for i, pt_idx in enumerate(order):
        if time.time() > deadline - 30:
            break
        
        if excluded[pt_idx]:
            continue
        
        # Try adding this point
        excluded[pt_idx] = True
        
        # Check validity - this is expensive, so only do it if we think it might work
        # Quick check: for each direction d, find points that share a line with pt_idx
        # and might lose direction d
        
        valid = True
        for d in range(num_dirs):
            # Find all points on the line through pt_idx in direction d
            # These are pt_idx itself and the others in line_points[d][pt_idx]
            # For each such point x (other than pt_idx), check if x still has direction d valid
            # Direction d is valid for x if no excluded point is on line through x in dir d (except x itself)
            # Since we just added pt_idx, if pt_idx is on the line through x in dir d,
            # and pt_idx != x, then direction d might no longer be valid for x
            
            for other_idx in line_points[d][pt_idx]:
                # Check if other_idx still has direction d valid
                # It's invalid if there's an excluded point on the line through other_idx in dir d
                # (other than other_idx itself)
                # We know pt_idx is on this line. Are there other excluded points?
                # We need to check all excluded points on this line.
                # This is expensive, so let's just do the full check for now
                pass
        
        # Full validity check (expensive but necessary)
        # Optimization: only check points that might be affected
        # Points affected: those on any line with pt_idx
        affected = set()
        for d in range(num_dirs):
            affected.add(pt_idx)
            for other_idx in line_points[d][pt_idx]:
                affected.add(other_idx)
        
        for x_idx in affected:
            if not check_point_valid(x_idx, excluded):
                valid = False
                break
        
        if valid:
            added_count += 1
        else:
            excluded[pt_idx] = False
    
    final_size = excluded.sum()
    print(f"Final excluded size: {final_size} (added {added_count})")

    # Build the point list
    points = []
    for pt_idx in range(total_points):
        if not excluded[pt_idx]:
            x_idx = pt_idx // q
            y_idx = pt_idx % q
            x_a = x_idx // p
            x_b = x_idx % p
            y_a = y_idx // p
            y_b = y_idx % p
            points.append([[x_a, x_b], [y_a, y_b]])

    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
