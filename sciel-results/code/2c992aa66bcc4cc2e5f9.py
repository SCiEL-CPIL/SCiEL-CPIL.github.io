import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 590.0

    p = 7
    q = p * p
    
    # Field operations for F_49
    # Represented as (a, b) for a + b*alpha, alpha^2 = 3
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)

    def mul(a, b):
        return ((a[0] * b[0] + a[1] * b[1] * 3) % p, (a[0] * b[1] + a[1] * b[0]) % p)

    def point_to_index(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb

    def index_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p * p)) % p
        xa = (idx // (p * p * p)) % p
        return (xa, xb, ya, yb)

    total_points = q * q
    
    # Generate S: Graph of y = x^2 in F_49
    # x = (xa, xb), y = x^2
    S_indices = set()
    
    for xa in range(p):
        for xb in range(p):
            x = (xa, xb)
            y = mul(x, x)
            ya, yb = y
            idx = point_to_index(xa, xb, ya, yb)
            S_indices.add(idx)
            
    # Verify validity
    # Precompute lines
    # Directions: (0, 1) [Vertical] and (1, m) for m in F_49
    lines = []
    point_to_lines = {i: [] for i in range(total_points)}
    
    def add_line(points_on_line):
        line_set = set(points_on_line)
        lines.append(line_set)
        line_id = len(lines) - 1
        for p_idx in line_set:
            point_to_lines[p_idx].append(line_id)
            
    # Direction (0, 1): x = c
    for ca in range(p):
        for cb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = point_to_index(ca, cb, ya, yb)
                    pts.append(idx)
            add_line(pts)
            
    # Direction (1, m): y = m*x + c
    for ma in range(p):
        for mb in range(p):
            m = (ma, mb)
            for ca in range(p):
                for cb in range(p):
                    c = (ca, cb)
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            x = (xa, xb)
                            mx = mul(m, x)
                            y = add(mx, c)
                            idx = point_to_index(xa, xb, y[0], y[1])
                            pts.append(idx)
                    add_line(pts)
                    
    valid = True
    for x_idx in range(total_points):
        in_S = x_idx in S_indices
        found_line = False
        for L_id in point_to_lines[x_idx]:
            L = lines[L_id]
            # Count points in L that are in S
            count_S = 0
            for p_idx in L:
                if p_idx in S_indices:
                    count_S += 1
                    if count_S > 1:
                        break
            
            if in_S:
                # Need L cap S = {x}
                if count_S == 1:
                    # Ensure x is the one
                    if x_idx in L: # Always true
                        found_line = True
                        break
            else:
                # Need L cap S = empty
                if count_S == 0:
                    found_line = True
                    break
        
        if not found_line:
            valid = False
            break
            
    if not valid:
        # Fallback to S = { (0,0) }
        S_indices = {point_to_index(0, 0, 0, 0)}
        
    # Construct N = F^2 \ S
    points = []
    for idx in range(total_points):
        if idx not in S_indices:
            xa, xb, ya, yb = index_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
