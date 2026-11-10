# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using a greedy set-cover approach
    to maximize the excluded set (complement).
    
    The idea: we want to select a set E (excluded) such that for every point x,
    there exists a direction d where all other 48 points on line(x, d) are in N (not in E).
    Equivalently, for every point x, there exists a direction d where no other point
    on line(x, d) is in E.
    
    We use a greedy approach: start with E empty, and greedily add points to E
    that "cover" the most points while maintaining validity.
    """
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    start_time = time.time()
    deadline = start_time + 600  # 600s limit
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Point index: idx = ((xa*p + xb)*p + ya)*p + yb
    def point_to_idx(xa, xb, ya, yb):
        return ((xa*p + xb)*p + ya)*p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return (xa, xb, ya, yb)
    
    # Directions: v = (1, m) for m in F_49, plus v = (0, 1)
    directions_v = []
    for ma in range(p):
        for mb in range(p):
            directions_v.append((1, 0, ma, mb))  # vx=(1,0), vy=(ma,mb)
    directions_v.append((0, 0, 0, 1))  # vx=(0,0), vy=(0,1)
    
    num_dirs = len(directions_v)  # 50
    
    # Precompute line membership: for each point and direction, the 48 other points
    line_other = np.zeros((total_points, num_dirs, q-1), dtype=np.int32)
    
    for idx in range(total_points):
        xa, xb, ya, yb = idx_to_point(idx)
        for d_idx, (vx_a, vx_b, vy_a, vy_b) in enumerate(directions_v):
            count = 0
            for ta in range(p):
                for tb in range(p):
                    if ta == 0 and tb == 0:
                        continue
                    tvx_a = (ta * vx_a + tb * vx_b * k) % p
                    tvx_b = (ta * vx_b + tb * vx_a) % p
                    tvy_a = (ta * vy_a + tb * vy_b * k) % p
                    tvy_b = (ta * vy_b + tb * vy_a) % p
                    nx_a = (xa + tvx_a) % p
                    nx_b = (xb + tvx_b) % p
                    ny_a = (ya + tvy_a) % p
                    ny_b = (yb + tvy_b) % p
                    n_idx = point_to_idx(nx_a, nx_b, ny_a, ny_b)
                    line_other[idx, d_idx, count] = n_idx
                    count += 1
    
    # For each point and direction, track whether the line is "clean" (no excluded points)
    # clean[point, dir] = True if no excluded point on line(point, dir) except possibly point itself
    
    # We'll use a different approach: model as a constraint satisfaction problem
    # and use local search with simulated annealing
    
    # Initialize with the baseline unital construction
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    excluded = np.zeros(total_points, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        idx = point_to_idx(xa, xb, ya, yb)
                        excluded[idx] = True
    
    # For each point, count how many directions are "clear" (all 48 others not excluded)
    def compute_clear_counts():
        clear = np.zeros(total_points, dtype=np.int32)
        for idx in range(total_points):
            for d_idx in range(num_dirs):
                others = line_other[idx, d_idx, :]
                if not np.any(excluded[others]):
                    clear[idx] += 1
        return clear
    
    clear_counts = compute_clear_counts()
    
    # Verify initial set is valid
    assert np.all(clear_counts >= 1), "Initial set is not a valid Nikodym set!"
    
    # Local search: try to add points to excluded
    # When adding point p, check if it maintains validity
    
    def can_add(p_idx):
        """Check if adding p_idx to excluded maintains validity."""
        # p_idx itself must have at least one clear direction
        p_has_clear = False
        for d_idx in range(num_dirs):
            others = line_other[p_idx, d_idx, :]
            if not np.any(excluded[others]):
                p_has_clear = True
                break
        if not p_has_clear:
            return False
        
        # Check all points on lines through p_idx
        for d_idx in range(num_dirs):
            others = line_other[p_idx, d_idx, :]
            for q_idx in others:
                # q_idx shares line with p_idx in direction d_idx
                # After adding p_idx, direction d_idx for q_idx may no longer be clear
                # Check if q_idx still has at least one clear direction
                q_clear = False
                for d2_idx in range(num_dirs):
                    q_others = line_other[q_idx, d2_idx, :]
                    # p_idx is on the line through q_idx in direction d2_idx iff
                    # p_idx is in line_other[q_idx, d2_idx, :] or p_idx == q_idx
                    # Since p_idx != q_idx, check if p_idx is in line_other[q_idx, d2_idx, :]
                    if d2_idx == d_idx:
                        # p_idx is definitely on this line (we're iterating over it)
                        # So this direction is no longer clear for q_idx
                        continue
                    else:
                        # Check if p_idx is on line(q_idx, d2_idx)
                        # This is equivalent to checking if q_idx is on line(p_idx, d2_idx)
                        # which we can check by seeing if q_idx is in line_other[p_idx, d2_idx, :]
                        if np.any(excluded[q_others]):
                            continue
                        # Also need to check if p_idx would be excluded and on this line
                        # p_idx is on line(q_idx, d2_idx) iff q_idx is on line(p_idx, d2_idx)
                        if q_idx in line_other[p_idx, d2_idx, :]:
                            continue
                        q_clear = True
                        break
                if not q_clear:
                    return False
        return True
    
    # Greedy local search
    for iteration in range(100):
        if time.time() > deadline - 30:
            break
        improved = False
        # Try to add each non-excluded point
        for p_idx in range(total_points):
            if time.time() > deadline - 30:
                break
            if not excluded[p_idx]:
                if can_add(p_idx):
                    excluded[p_idx] = True
                    improved = True
        if not improved:
            break
    
    # Final verification
    clear_counts = compute_clear_counts()
    if not np.all(clear_counts >= 1):
        # Fallback to baseline
        excluded = np.zeros(total_points, dtype=bool)
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            idx = point_to_idx(xa, xb, ya, yb)
                            excluded[idx] = True
    
    # Build the result: N = all points NOT in excluded
    points = []
    for idx in range(total_points):
        if not excluded[idx]:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
