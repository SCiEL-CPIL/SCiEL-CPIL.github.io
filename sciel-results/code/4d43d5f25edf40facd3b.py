import numpy as np
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2.
    """
    p = 7
    k = None
    for c in range(2, p):
        if pow(c, (p - 1) // 2, p) == p - 1:
            k = c
            break

    # Generate all points on the algebraic surface
    surface_points = set()
    for ya in range(p):
        for yb in range(p):
            rhs = (ya * ya - k * yb * yb) % p
            inv2 = pow(2, p - 2, p)
            xa = (rhs * inv2) % p
            for xb in range(p):
                surface_points.add((xa, xb, ya, yb))

    # Try different exclusion patterns and find the largest valid one
    best_N = None
    best_complement = -1

    # Try excluding based on quadratic forms in y
    for c1 in range(p):
        for c2 in range(p):
            if c1 == 0 and c2 == 0:
                continue
            
            excluded_y = set()
            for ya in range(p):
                for yb in range(p):
                    if (c1 * ya * ya + c2 * yb * yb) % p == 0:
                        excluded_y.add((ya, yb))
            
            excluded = set()
            for (xa, xb, ya, yb) in surface_points:
                if (ya, yb) in excluded_y:
                    excluded.add((xa, xb, ya, yb))
            
            # Build N
            N_set = set()
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            if (xa, xb, ya, yb) not in excluded:
                                N_set.add((xa, xb, ya, yb))
            
            # Quick validity check: sample some points
            valid = True
            sample_points = list(N_set)[:100]  # Check first 100 points
            for (xa, xb, ya, yb) in sample_points:
                # Check if there's a valid line through this point
                found_valid_line = False
                # Try all slopes m
                for ma in range(p):
                    for mb in range(p):
                        # Direction (1, m) where m = ma + mb*alpha
                        # Check if all other points on line are in N
                        line_ok = True
                        for t in range(p):
                            if t == 0:
                                continue
                            # x' = x + t*1, y' = y + t*m
                            xa_t = (xa + t) % p
                            xb_t = (xb + 0) % p  # 1 = (1,0)
                            # y' = y + t*m
                            ya_t = (ya + t * ma) % p
                            yb_t = (yb + t * mb) % p
                            if (xa_t, xb_t, ya_t, yb_t) not in N_set:
                                line_ok = False
                                break
                        if line_ok:
                            found_valid_line = True
                            break
                    if found_valid_line:
                        break
                
                if not found_valid_line:
                    # Try vertical line (0,1)
                    line_ok = True
                    for t in range(p):
                        if t == 0:
                            continue
                        xa_t = xa
                        xb_t = xb
                        ya_t = (ya + t) % p
                        yb_t = yb
                        if (xa_t, xb_t, ya_t, yb_t) not in N_set:
                            line_ok = False
                            break
                    if line_ok:
                        found_valid_line = True
                
                if not found_valid_line:
                    valid = False
                    break
            
            if valid and len(N_set) > best_complement:
                best_complement = len(N_set)
                best_N = N_set

    if best_N is None:
        # Fallback to baseline
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))

        points = []
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
                        in_y_special = (ya, yb) in y_special
                        if in_unital and not in_y_special:
                            continue
                        points.append([[xa, xb], [ya, yb]])
        return np.array(points, dtype=np.int64)

    N_list = []
    for (xa, xb, ya, yb) in sorted(best_N):
        N_list.append([[xa, xb], [ya, yb]])

    return np.array(N_list, dtype=np.int64)
