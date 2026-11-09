import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field operations for F_49 = F_7[alpha], alpha^2 = 3
    k = 3
    def f_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    def f_sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    def f_mul(a, b):
        # (a0 + a1*alpha) * (b0 + b1*alpha) = a0b0 + (a0b1 + a1b0)alpha + a1b1*alpha^2
        # = (a0b0 + a1b1*k) + (a0b1 + a1b0)alpha
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    def f_neg(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    # Precompute all points
    points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    points.append(((xa, xb), (ya, yb)))
                    
    # Precompute lines? 
    # There are q*(q+1) lines.
    # Each line has q points.
    # We can represent a line by (point, slope) or (point, vertical).
    
    # Construct Hermitian Unital U
    # Trace(x) + Norm(y) = 0
    # Trace(x) = x + x^p = x + x^bar. In F_p[alpha], x=(a,b), x^p=(a, -b) since alpha^p = -alpha.
    # So Trace(x) = 2a.
    # Norm(y) = y * y^p = (ya, yb)(ya, -yb) = (ya^2 - yb^2 k, 0).
    # So condition: 2*xa + (ya^2 - yb^2 k) = 0 mod p.
    
    U = set()
    for pt in points:
        (xa, xb), (ya, yb) = pt
        if (2*xa + ya*ya - yb*yb*k) % p == 0:
            U.add(pt)
            
    S = set(U)
    
    # Helper to check if a point x is "bad"
    # x is bad if for ALL directions v, the line L_x(v) intersects S.
    # Directions: slopes m in F_q and vertical.
    # We can iterate over all lines through x.
    
    # Precompute lines through each point?
    # Or just generate them on the fly.
    
    # To speed up, we can maintain for each point x the number of "empty" lines (lines through x disjoint from S).
    # Initially, for x in U, empty lines = 0? No, we only care about x NOT in S.
    # For x not in S, we need at least 1 line disjoint from S.
    
    # Let's just do the greedy loop.
    # Max iterations: |S|
    
    start_time = time.time()
    deadline = start_time + 500 # Leave 100s margin
    
    while time.time() < deadline:
        # Find a bad point
        bad_point = None
        for x in points:
            if x in S:
                continue
            # Check if there is a line through x disjoint from S
            found_empty = False
            # Check vertical line
            # Line: {(x_a, t_b) : t_b in F_p}? No, vertical line in F_q^2 is {(x, y) : x = const}.
            # Wait, coordinates are (x, y) in F_q^2.
            # Vertical line: x is constant. y varies.
            # Points: (x, y') for all y' in F_q.
            
            # Check if vertical line is disjoint from S
            vert_disjoint = True
            for y in range(p):
                for yb in range(p):
                    if ((x[0], x[1]), (y, yb)) in S:
                        vert_disjoint = False
                        break
                if not vert_disjoint: break
            
            if vert_disjoint:
                found_empty = True
            else:
                # Check slopes m in F_q
                for ma in range(p):
                    for mb in range(p):
                        # Line: y = m*x + c? No, through x.
                        # y - x_y = m * (x_coord - x_x) ?
                        # In vector form: P = x + t * (1, m) for t in F_q.
                        # Or P = x + t * (u, m) where u is fixed?
                        # Direction v = (1, m) or (0, 1).
                        # We already checked (0,1).
                        # Now check (1, m) for all m in F_q.
                        
                        disjoint = True
                        for ta in range(p):
                            for tb in range(p):
                                t = (ta, tb)
                                # P = x + t*(1, m)
                                # x_coord + t*1
                                # y_coord + t*m
                                tx = f_add(x[0], t)
                                ty = f_add(x[1], f_mul(t, (ma, mb)))
                                if (tx, ty) in S:
                                    disjoint = False
                                    break
                            if not disjoint: break
                        if disjoint:
                            found_empty = True
                            break
                if found_empty:
                    break
            
            if not found_empty:
                bad_point = x
                break
                
        if bad_point is None:
            break
            
        # We found a bad point x.
        # We need to remove a point from S to make a line through x disjoint.
        # Try all lines through x, find one with min intersection with S.
        min_count = q + 1
        best_line_points = []
        
        # Vertical
        v_pts = []
        count = 0
        for ya in range(p):
            for yb in range(p):
                pt = ((x[0], x[1]), (ya, yb))
                v_pts.append(pt)
                if pt in S:
                    count += 1
        if count < min_count:
            min_count = count
            best_line_points = v_pts
            
        # Slopes
        for ma in range(p):
            for mb in range(p):
                l_pts = []
                count = 0
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        tx = f_add(x[0], t)
                        ty = f_add(x[1], f_mul(t, (ma, mb)))
                        pt = (tx, ty)
                        l_pts.append(pt)
                        if pt in S:
                            count += 1
                if count < min_count:
                    min_count = count
                    best_line_points = l_pts
                    
        # Remove a point from best_line_points that is in S
        # To be smart, remove a point that is in S and appears in many other "bad" lines?
        # For now, just pick the first one.
        for pt in best_line_points:
            if pt in S:
                S.remove(pt)
                break
                
    # Construct N
    N = []
    for pt in points:
        if pt not in S:
            N.append(pt)
            
    return np.array(N, dtype=np.int64)
