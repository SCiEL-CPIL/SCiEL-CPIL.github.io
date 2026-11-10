# EVOLVE-BLOCK-START
import numpy as np
import time


def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_q^2 with q = p^2 and p = 7. A set N is a Nikodym
    set if, for every point x in F_q^2, there exists a direction v != 0 such that
    the punctured line {x + t*v : t in F_q, t != 0} is fully contained in N. The
    goal is to make N as SMALL as possible, equivalently to make its complement
    (the excluded set) as LARGE as possible.

    Field model: F_q is built as F_p[alpha] with alpha^2 = k, where k is the
    smallest quadratic non-residue mod p. An element of F_q is the pair (a, b)
    standing for a + b*alpha, with a, b in {0, ..., p-1}.

    Construction: Take the Hermitian unital
        U = {(x, y) : Trace(x) + Norm(y) = 0},  with Trace((a,b)) = 2a,
        Norm((a,b)) = a^2 - b^2*k.
    Build the excluded set S = points of U whose y-coordinate lies in the special
    set Y = {(u^2 mod p, v) : u, v in F_p, u != 0}. The Nikodym set is N = (F_q^2) \\ S.

    Returns:
        points: np.ndarray of shape (k, 2, 2) integer array. Each of the k points
        is [[x_a, x_b], [y_a, y_b]] with the F_q coordinates x = (x_a, x_b) and
        y = (y_a, y_b), all entries in [0, p).
    """
    p = 7
    start_time = time.time()
    deadline = 600.0  # seconds

    # Smallest quadratic non-residue mod p (the alpha^2 = k constant).
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Special y-coordinate set Y: y_a must be a non-zero square mod p.
    y_special = set()
    for u in range(1, p):
        for v in range(p):
            y_special.add(((u * u) % p, v))

    # Try to expand the excluded set by also excluding points where x is in a special set
    # and checking if the Nikodym property still holds.
    
    # First, let's compute the base excluded set S
    base_excluded = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    if in_unital and not in_y_special:
                        base_excluded.add((xa, xb, ya, yb))
    
    # Now try to add more points to S while maintaining the Nikodym property.
    # A point can be added to S if, for every line through it in any of the 50 directions,
    # at most one other point on that line is in S.
    
    # Precompute all lines for each direction
    # Direction (1, m) where m = (ma, mb) in F_49, and direction (0, 1)
    
    # For efficiency, let's check if we can add points from the unital where y IS in y_special
    # but x is in some special set.
    
    # Let's try a different approach: exclude points from U where either y is not in Y
    # OR x is not in X, for some special set X.
    
    # Define X as points where x_a is a non-zero square
    x_special = set()
    for u in range(1, p):
        for v in range(p):
            x_special.add(((u * u) % p, v))
    
    # Try excluding points where (y not in Y) OR (x not in X)
    # But this might be too aggressive. Let's try: exclude if (y not in Y) AND (x not in X)
    # Wait, that would be smaller. Let's try: exclude if (y not in Y) OR (x not in X)
    
    # Actually, let's be more careful. The Nikodym property requires that for each point x,
    # there exists a direction v such that all other points on the line are in N.
    # If we exclude too many points, we might violate this.
    
    # Let's try: exclude points from U where y is not in Y (as before), 
    # PLUS points from U where x is not in X and y is in Y.
    # This is: (in_unital AND (not in_y_special OR not in_x_special))
    
    # But we need to verify the Nikodym property. Let's implement a checker.
    
    def is_valid_nikodym(excluded_set):
        """Check if the complement of excluded_set is a valid Nikodym set."""
        # For each point in F_49^2, check if there exists a direction v such that
        # all other points on the line through the point with direction v are in N.
        
        # Precompute for each point and each direction, whether the line is "good"
        # (all other points on the line are in N)
        
        # Directions: (1, m) for m in F_49, and (0, 1)
        # Total: 49 + 1 = 50 directions
        
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        point = (xa, xb, ya, yb)
                        
                        # Check if this point is in N (not excluded)
                        # If it's excluded, we still need to check the Nikodym property
                        # (the property must hold for ALL points, not just those in N)
                        
                        found_good_direction = False
                        
                        # Try direction (0, 1): vertical line
                        # Line: (xa, xb, ya + t, yb) for t in F_7
                        # Wait, direction (0, 1) means v = (0, 1) in F_49^2
                        # So the line is: (xa + t*0, xb + t*0, ya + t*1, yb + t*0)
                        # = (xa, xb, ya + t, yb) for t in F_7
                        # But t is in F_49, not F_7!
                        
                        # I made an error. The line is {x + t*v : t in F_49}
                        # where t ranges over all 49 elements of F_49.
                        
                        # For direction (0, 1): v = (0, 1)
                        # Line: (xa, xb, ya + t, yb) where t in F_49
                        # t = (ta, tb) represents ta + tb*alpha
                        # ya + t = (ya + ta, yb + tb)  [since ya, yb are components]
                        # Wait, ya is a single component. Let me reclarify.
                        
                        # A point is (x, y) where x = (xa, xb) and y = (ya, yb) in F_49.
                        # Direction v = (vx, vy) where vx, vy in F_49.
                        # Line: {(x + t*vx, y + t*vy) : t in F_49}
                        
                        # For v = (0, 1): vx = (0, 0), vy = (1, 0)
                        # Line: {(x, y + t*(1,0)) : t in F_49}
                        # = {(x, (ya + ta, yb + tb)) : ta, tb in F_7}
                        # Since t = (ta, tb) ranges over all F_49, the line is:
                        # {(xa, xb, ya + ta, yb + tb) : ta, tb in F_7}
                        # This is a 49-point line.
                        
                        # Check if all points on this line except the given point are in N
                        good = True
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0:
                                    continue  # skip the point itself
                                ny = (ya + ta) % p
                                nz = (yb + tb) % p
                                if (xa, xb, ny, nz) in excluded_set:
                                    good = False
                                    break
                            if not good:
                                break
                        if good:
                            found_good_direction = True
                        else:
                            # Try other directions
                            pass
                        
                        if found_good_direction:
                            continue
                        
                        # Try directions (1, m) for m in F_49
                        # v = (1, m) where 1 = (1, 0) in F_49
                        # Line: {(x + t, y + t*m) : t in F_49}
                        # x + t = (xa + ta, xb + tb)
                        # y + t*m = (ya + ta*ma - tb*mb*k, yb + ta*mb + tb*ma)
                        # where m = (ma, mb)
                        
                        for ma in range(p):
                            for mb in range(p):
                                good = True
                                for ta in range(p):
                                    for tb in range(p):
                                        if ta == 0 and tb == 0:
                                            continue
                                        nx_a = (xa + ta) % p
                                        nx_b = (xb + tb) % p
                                        ny_a = (ya + ta * ma - tb * mb * k) % p
                                        ny_b = (yb + ta * mb + tb * ma) % p
                                        if (nx_a, nx_b, ny_a, ny_b) in excluded_set:
                                            good = False
                                            break
                                    if not good:
                                        break
                                if good:
                                    found_good_direction = True
                                    break
                            if found_good_direction:
                                break
                        
                        if not found_good_direction:
                            return False, point
        
        return True, None
    
    # The checker above is O(2401 * 50 * 49) = ~5.9M operations, which should be fast.
    # But let's optimize by precomputing.
    
    # Actually, let's just try a few candidate constructions and check them.
    
    # Candidate 1: Original (exclude if in_unital AND not in_y_special)
    # Candidate 2: Exclude if in_unital AND (not in_y_special OR not in_x_special)
    # Candidate 3: Exclude if in_unital AND not in_y_special (original)
    
    # Let's try candidate 2
    excluded_set = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                    in_y_special = (ya, yb) in y_special
                    in_x_special = (xa, xb) in x_special
                    if in_unital and (not in_y_special or not in_x_special):
                        excluded_set.add((xa, xb, ya, yb))
    
    # Check if this is valid
    # For speed, let's implement a faster checker
    def check_nikodym_fast(excluded_set):
        """Fast check using precomputed line membership."""
        # For each point and each direction, count how many other points on the line are excluded
        # A point is "good" if there exists a direction where 0 other points are excluded
        
        # Precompute excluded as a 4D boolean array
        excluded = np.zeros((p, p, p, p), dtype=bool)
        for (xa, xb, ya, yb) in excluded_set:
            excluded[xa, xb, ya, yb] = True
        
        # For each point, check all 50 directions
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        # Direction (0, 1): line is (xa, xb, ya+ta, yb+tb)
                        # Count excluded points on this line (excluding the point itself)
                        count = 0
                        for ta in range(p):
                            for tb in range(p):
                                if ta == 0 and tb == 0:
                                    continue
                                if excluded[xa, xb, (ya+ta)%p, (yb+tb)%p]:
                                    count += 1
                        if count == 0:
                            continue  # good direction found
                        
                        # Directions (1, m)
                        found = False
                        for ma in range(p):
                            for mb in range(p):
                                count = 0
                                for ta in range(p):
                                    for tb in range(p):
                                        if ta == 0 and tb == 0:
                                            continue
                                        nx_a = (xa + ta) % p
                                        nx_b = (xb + tb) % p
                                        ny_a = (ya + ta*ma - tb*mb*k) % p
                                        ny_b = (yb + ta*mb + tb*ma) % p
                                        if excluded[nx_a, nx_b, ny_a, ny_b]:
                                            count += 1
                                if count == 0:
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            return False
        return True
    
    # The fast checker is still O(2401 * 50 * 49) = ~5.9M, which is fine.
    # But let's check the time
    if time.time() - start_time > deadline - 30:
        # Not enough time, return the original
        pass
    
    valid = check_nikodym_fast(excluded_set)
    
    if valid:
        # Great, we have a larger excluded set
        pass
    else:
        # Fall back to original
        excluded_set = base_excluded
    
    # Build the Nikodym set (complement of excluded_set)
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (xa, xb, ya, yb) not in excluded_set:
                        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)


# EVOLVE-BLOCK-END
