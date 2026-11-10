import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    M = q * q
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # Element is (a, b) -> a + b*alpha
    # We map (a, b) to integer idx = a * p + b
    # Point is (x, y) where x, y in F_49
    # Point index = x_idx * q + y_idx
    
    # Precompute lines
    # A line is defined by point (x0, y0) and slope m (or vertical)
    # There are q+1 = 50 directions.
    # Directions: 0, 1, ..., p-1, inf (vertical)
    # In F_49, slopes are elements of F_49.
    # Vertical line: x = c.
    # Slope m line: y - y0 = m * (x - x0).
    
    # We need to iterate through points and lines efficiently.
    # Let's represent points as indices 0..M-1.
    # x_idx = pt // q, y_idx = pt % q.
    
    # Directions:
    # 0..48: slope m = (a, b)
    # 49: vertical
    
    # For each point, list of lines passing through it.
    # Total lines = q * (q+1) = 49 * 50 = 2450.
    # Each line has q points.
    
    # Let's generate all lines.
    # Line ID: 0..2449
    # For each line, store the set of points on it.
    # And for each point, store the list of line IDs.
    
    lines = []
    point_lines = [[] for _ in range(M)]
    
    line_id = 0
    # Vertical lines: x = c
    for c in range(q):
        pts = []
        for y in range(q):
            pt = c * q + y
            pts.append(pt)
            point_lines[pt].append(line_id)
        lines.append(pts)
        line_id += 1
        
    # Sloped lines: y = m*x + b
    # m in F_49, b in F_49
    # y - b = m * x
    # (ya, yb) - (ba, bb) = m * (xa, xb)
    # This requires field arithmetic.
    
    # Field ops for F_7[alpha], alpha^2 = 3
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    def mul_f(a, b):
        # (a0 + a1*a) * (b0 + b1*a) = a0b0 + a0b1a + a1b0a + a1b1a^2
        # = (a0b0 + 3*a1b1) + (a0b1 + a1b0)a
        return ((a[0]*b[0] + 3*a[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    def neg_f(a):
        return ((-a[0]) % p, (-a[1]) % p)
        
    # Precompute multiplication table for speed?
    # mul_table[m_idx][x_idx] = y_idx
    # m_idx 0..48
    # x_idx 0..48
    mul_table = [[0]*q for _ in range(q)]
    add_table = [[0]*q for _ in range(q)]
    sub_table = [[0]*q for _ in range(q)]
    
    for i in range(q):
        a = (i // p, i % p)
        for j in range(q):
            b = (j // p, j % p)
            mul_table[i][j] = mul_f(a, b)[0]*p + mul_f(a, b)[1]
            add_table[i][j] = add_f(a, b)[0]*p + add_f(a, b)[1]
            sub_table[i][j] = sub_f(a, b)[0]*p + sub_f(a, b)[1]

    for m_idx in range(q):
        for b_idx in range(q):
            pts = []
            for x_idx in range(q):
                # y = m*x + b
                y_idx = add_table[mul_table[m_idx][x_idx]][b_idx]
                pt = x_idx * q + y_idx
                pts.append(pt)
                point_lines[pt].append(line_id)
            lines.append(pts)
            line_id += 1
            
    # line_id should be 2450
    
    # Greedy Algorithm
    start_time = time.time()
    best_S = set()
    
    # Try multiple random orders
    for restart in range(10):
        if time.time() - start_time > 500:
            break
            
        S = set()
        n_L = [0] * len(lines)
        # C0[y] = count of lines through y with n_L = 0
        # C1[y] = count of lines through y with n_L = 1
        C0 = [len(point_lines[y]) for y in range(M)] # Initially all n_L=0
        C1 = [0] * M
        
        # Random order of points
        order = list(range(M))
        np.random.shuffle(order)
        
        for x in order:
            # Check if x can be added
            # 1. C0[x] > 0
            if C0[x] == 0:
                continue
                
            # 2. Check constraints for all y
            valid = True
            # We need to check if any y loses its last valid line.
            # For y in S: valid lines are those with n_L=1.
            # For y not in S: valid lines are those with n_L=0.
            
            # We only need to check y that are on lines through x.
            # Let affected_lines = point_lines[x]
            # For each L in affected_lines:
            #   old_n = n_L[L]
            #   new_n = old_n + 1
            #   For each y in lines[L]:
            #      if y == x: continue (handled by C0[x]>0)
            #      if y in S:
            #         if old_n == 1:
            #            C1[y] decreases by 1.
            #            if C1[y] == 1 (before dec, so 0 after): invalid
            #      else:
            #         if old_n == 0:
            #            C0[y] decreases by 1.
            #            if C0[y] == 1 (before dec, so 0 after): invalid
            
            # Optimization:
            # If C1[y] > 1 for all y in S on affected lines, and C0[y] > 1 for all y not in S on affected lines, then valid.
            # Actually, we just need to check if ANY y drops to 0.
            
            # To speed up, we can pre-check.
            # But M is small. Let's just iterate.
            
            # Collect unique affected points
            # affected_pts = set()
            # for L in point_lines[x]:
            #    for y in lines[L]:
            #       if y != x: affected_pts.add(y)
            
            # This is 50 * 49 = 2450 points.
            # Checking 2450 points per candidate x.
            # If we try many candidates, it's slow.
            
            # Better:
            # Just check the points on the lines through x.
            
            # We can break early if invalid.
            
            # Let's do the check.
            # We need to be careful not to double count if a point is on multiple lines through x?
            # A point y (y!=x) is on exactly ONE line through x.
            # So no double counting.
            
            for L_idx in point_lines[x]:
                L = lines[L_idx]
                old_n = n_L[L_idx]
                
                for y in L:
                    if y == x:
                        continue
                    
                    if y in S:
                        # y needs n_L=1
                        if old_n == 1:
                            # This line L will become n_L=2, so no longer valid for y
                            # y loses one valid line.
                            # If C1[y] was 1, it becomes 0 -> Invalid
                            if C1[y] == 1:
                                valid = False
                                break
                    else:
                        # y needs n_L=0
                        if old_n == 0:
                            # This line L will become n_L=1, so no longer valid for y
                            # y loses one valid line.
                            # If C0[y] was 1, it becomes 0 -> Invalid
                            if C0[y] == 1:
                                valid = False
                                break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Add x to S
            S.add(x)
            
            # Update n_L, C0, C1
            for L_idx in point_lines[x]:
                L = lines[L_idx]
                old_n = n_L[L_idx]
                n_L[L_idx] = old_n + 1
                new_n = old_n + 1
                
                for y in L:
                    if old_n == 0:
                        # n_L went 0 -> 1
                        C0[y] -= 1
                        C1[y] += 1
                    elif old_n == 1:
                        # n_L went 1 -> 2
                        C1[y] -= 1
                        # C0[y] unchanged
                    # else: no change to C0, C1
                    
            # End of step
            
        if len(S) > len(best_S):
            best_S = S.copy()
            
    # Construct N = F_49^2 \ S
    # Return points of N
    points = []
    for pt in range(M):
        if pt not in best_S:
            x_idx = pt // q
            y_idx = pt % q
            xa = x_idx // p
            xb = x_idx % p
            ya = y_idx // p
            yb = y_idx % p
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
