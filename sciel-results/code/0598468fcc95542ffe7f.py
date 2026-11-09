# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 with q = p^2 and p = 7.
    Uses the Hermitian unital construction and attempts to extend the complement
    by checking if additional points can be validly excluded.
    """
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    p = 7
    q = p * p  # 49
    total_points = q * q  # 2401
    
    # Smallest quadratic non-residue mod p
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break
    
    # Special y-coordinate set Y = {(u^2 mod p, v)}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    # Build the unital U: points where 2*xa + ya^2 - yb^2*k ≡ 0 (mod 7)
    unital_points = set()
    for ya in range(p):
        for yb in range(p):
            rhs = (-(ya * ya - yb * yb * k)) % p
            xa = (rhs * pow(2, p - 2, p)) % p
            for xb in range(p):
                unital_points.add((xa, xb, ya, yb))
    
    # Initial excluded set: unital points NOT in Y
    excluded = set()
    for pt in unital_points:
        xa, xb, ya, yb = pt
        if (ya, yb) not in y_special:
            excluded.add(pt)
    
    # N = all points not in excluded
    N = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    pt = (xa, xb, ya, yb)
                    if pt not in excluded:
                        N.add(pt)
    
    # Precompute for each point, the set of other points on each line through it
    # This is expensive but we only need to do it once
    
    # For each point and direction, compute the line's other points
    # Directions: v = (0,1) and v = (1, m) for m in F_49 (49 values)
    # Total directions: 50
    
    # Precompute line membership: for each direction, group points into lines
    # Then for a point to be removable, there must exist a direction where all other
    # points on its line are in N
    
    # Let's precompute for each point, for each direction, the count of N-points on the line
    # If count == 48 (all other points), then the point can be removed
    
    # For direction v = (0, 1): lines are vertical, each has 49 points
    # For direction v = (1, m): lines are "slanted", each has 49 points
    
    # Precompute line assignments
    # For v = (0,1): line through (xa,xb,ya,yb) contains all (xa,xb,ya',yb')
    # For v = (1,m) where m=(ma,mb): line through (xa,xb,ya,yb) contains
    #   (xa+ta, xb+tb, ya + (ta*ma+tb*mb*k), yb + (ta*mb+tb*ma)) for all t
    
    # This is getting complex. Let me try a different approach:
    # For each point, check if it can be removed by testing each direction
    
    # To speed up, precompute for each direction and each point, whether all other
    # points on the line are in N. We can do this incrementally.
    
    # Alternative: use the fact that we're only trying to remove a few points.
    # Start with the initial N, and try to remove points one at a time.
    
    # For efficiency, let's precompute the lines once
    
    # For direction (0,1):
    # Line through (xa,xb,ya,yb) = {(xa,xb,ya',yb') for all ya',yb'}
    
    # For direction (1, m) with m=(ma,mb):
    # Line through (xa,xb,ya,yb) = {(xa+ta, xb+tb, ya + (ta*ma+tb*mb*k)%p, yb + (ta*mb+tb*ma)%p) for ta,tb in 0..6}
    
    # Precompute for each direction, a mapping from point to line_id, and line_id to set of points
    
    directions = [(0, 1)]  # v = (0,1)
    for ma in range(p):
        for mb in range(p):
            directions.append((ma, mb))  # v = (1, m)
    
    # For each direction, compute the line structure
    # line_struct[dir_idx] = dict mapping point -> set of other points on same line
    
    # This is memory intensive: 2401 points * 50 directions * 48 points = ~5.7M entries
    # Let's try a different approach: for each point, store the count of N-members on each line
    
    # Actually, let's just do the greedy removal with efficient checks
    
    def can_remove_fast(pt, N_set, p, k):
        xa, xb, ya, yb = pt
        
        # Check v = (0,1): vertical line
        # All other points: (xa, xb, ya', yb') for all (ya',yb') != (ya,yb)
        for ya2 in range(p):
            for yb2 in range(p):
                if ya2 == ya and yb2 == yb:
                    continue
                if (xa, xb, ya2, yb2) not in N_set:
                    break
            else:
                continue
            break
        else:
            return True
        
        # Check v = (1, m) for each m
        for ma in range(p):
            for mb in range(p):
                found_bad = False
                for ta in range(p):
                    for tb in range(p):
                        if ta == 0 and tb == 0:
                            continue
                        nx_a = (xa + ta) % p
                        nx_b = (xb + tb) % p
                        tm_a = (ta * ma + tb * mb * k) % p
                        tm_b = (ta * mb + tb * ma) % p
                        ny_a = (ya + tm_a) % p
                        ny_b = (yb + tm_b) % p
                        if (nx_a, nx_b, ny_a, ny_b) not in N_set:
                            found_bad = True
                            break
                    if found_bad:
                        break
                if not found_bad:
                    return True
        
        return False
    
    # Try to remove points, prioritizing unital points in N
    candidates = [pt for pt in N if pt in unital_points] + [pt for pt in N if pt not in unital_points]
    
    removed_count = 0
    for pt in candidates:
        if time.time() > deadline - 60:
            break
        
        if pt not in N:
            continue
        
        if can_remove_fast(pt, N, p, k):
            N.discard(pt)
            excluded.add(pt)
            removed_count += 1
    
    # Build the result array
    points = list(N)
    result = np.zeros((len(points), 2, 2), dtype=np.int64)
    for i, (xa, xb, ya, yb) in enumerate(points):
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
    
    return result


# EVOLVE-BLOCK-END
