import numpy as np

def nikodym_set():
    # Field F_49 = F_7[alpha] with alpha^2 = 3
    # Points are [[x_a, x_b], [y_a, y_b]]
    
    # We construct S (N) as the complement of C.
    # C is a set of 48 points on the line y=0, excluding the origin.
    # C = { (x, 0) | x in F_49, x != 0 }
    # This C does not contain a full line (size 49), so for any x not in C,
    # there is at least one line through x disjoint from C.
    # For points in C, the tangent/other lines work as witnesses.
    # Specifically, for x in C (on y=0, x!=0), the line x=const (vertical) 
    # intersects C only at x? No, vertical line x=c contains (c,0) which is in C.
    # It contains no other points of C because C is on y=0.
    # So the vertical line through x is a valid witness for x in C.
    # For x in S:
    # If x is on y=0 but x=0 (origin), lines through origin:
    # Line y=0 contains C. Blocked.
    # Other 48 lines through origin intersect y=0 only at origin.
    # Origin is not in C. So these lines are disjoint from C. Valid.
    # If x is not on y=0, lines to C are distinct. 48 blocked, 1 free. Valid.
    
    points = []
    for xa in range(7):
        for xb in range(7):
            for ya in range(7):
                for yb in range(7):
                    # Check if point is in C
                    # C: y=0 and x != 0
                    is_y_zero = (ya == 0 and yb == 0)
                    is_x_zero = (xa == 0 and xb == 0)
                    
                    if is_y_zero and not is_x_zero:
                        # Point is in C, so NOT in N
                        continue
                    else:
                        # Point is in N
                        points.append([[xa, xb], [ya, yb]])
                        
    return np.array(points, dtype=np.int64)
