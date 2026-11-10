import numpy as np
import time

def nikodym_set() -> np.ndarray:
    start_time = time.time()
    DEADLINE = 600  # seconds
    p = 7
    q = 49
    k = 3  # alpha^2 = 3
    
    # Precompute field operations using integer indices 0..48
    # idx = a * p + b represents a + b*alpha
    # Precompute addition, subtraction, multiplication tables
    add_tbl = np.zeros((q, q), dtype=np.int64)
    sub_tbl = np.zeros((q, q), dtype=np.int64)
    mul_tbl = np.zeros((q, q), dtype=np.int64)
    inv_tbl = np.zeros(q, dtype=np.int64)
    
    for i in range(q):
        a0, a1 = i // p, i % p
        for j in range(q):
            b0, b1 = j // p, j % p
            # Addition
            r0, r1 = (a0 + b0) % p, (a1 + b1) % p
            add_tbl[i, j] = r0 * p + r1
            # Subtraction
            r0, r1 = (a0 - b0) % p, (a1 - b1) % p
            sub_tbl[i, j] = r0 * p + r1
            # Multiplication
            r0 = (a0 * b0 + a1 * b1 * k) % p
            r1 = (a0 * b1 + a1 * b0) % p
            mul_tbl[i, j] = r0 * p + r1
        
        # Inverse
        if i != 0:
            denom = (a0 * a0 - a1 * a1 * k) % p
            if denom != 0:
                d_inv = pow(denom, p - 2, p)
                # (a0 - a1*alpha) * d_inv
                r0 = (a0 * d_inv) % p
                r1 = ((-a1) % p * d_inv) % p
                inv_tbl[i] = r0 * p + r1
            else:
                inv_tbl[i] = 0  # should not happen for non-zero
    
    # Points in F_49^2: index = x * q + y where x, y in 0..48
    total_points = q * q  # 2401
    
    # Construct Unital U: 2*xa + ya^2 - k*yb^2 = 0
    U = set()
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - k * yb * yb) % p == 0:
                        x_idx = xa * p + xb
                        y_idx = ya * p + yb
                        U.add(x_idx * q + y_idx)
    
    all_points_set = set(range(total_points))
    outside_U = all_points_set - U
    
    # Build lines
    # Vertical lines: x = c, direction (0,1)
    # Slanted lines: y = m*x + b, direction (1,m)
    lines = []
    
    # Vertical lines: 49 lines, each with 49 points
    for c in range(q):
        pts = []
        for y in range(q):
            pts.append(c * q + y)
        lines.append(pts)
    
    # Slanted lines: y = m*x + b
    for m in range(q):
        for b in range(q):
            pts = []
            for x in range(q):
                y = add_tbl[mul_tbl[m, x], b]
                pts.append(x * q + y)
            lines.append(pts)
    
    # For each line, compute intersection with U and coverage of outside_U
    line_U_intersections = []
    line_outside_coverage = []
    
    for L in lines:
        I = set()
        cov = set()
        for pt in L:
            if pt in U:
                I.add(pt)
            else:
                cov.add(pt)
        line_U_intersections.append(I)
        line_outside_coverage.append(cov)
    
    # Greedy Set Cover for outside_U
    uncovered = set(outside_U)
    H = set()
    
    # Track which lines are still useful
    # We'll use a priority approach: repeatedly pick the line with best score
    
    # For efficiency, maintain uncovered as a set
    # Score = len(actual_new_cov) / (len(I_L - H) + 1)
    
    max_iterations = 5000
    iteration = 0
    
    while uncovered and iteration < max_iterations:
        if time.time() - start_time > DEADLINE - 30:
            break
            
        best_line_idx = -1
        best_score = -1
        
        for i in range(len(lines)):
            if not line_outside_coverage[i]:
                continue
            
            # Compute actual new coverage
            actual_new_cov = 0
            for pt in line_outside_coverage[i]:
                if pt in uncovered:
                    actual_new_cov += 1
            
            if actual_new_cov == 0:
                continue
            
            I_L = line_U_intersections[i]
            new_H_cost = 0
            for pt in I_L:
                if pt not in H:
                    new_H_cost += 1
            
            score = actual_new_cov / (new_H_cost + 1)
            
            if score > best_score:
                best_score = score
                best_line_idx = i
        
        if best_line_idx == -1:
            break
        
        # Pick best line
        I_L = line_U_intersections[best_line_idx]
        H.update(I_L)
        
        # Update uncovered
        for pt in lines[best_line_idx]:
            if pt in uncovered:
                uncovered.remove(pt)
        
        iteration += 1
    
    # H is the set of points in U that are in N.
    # C = U - H
    C = U - H
    
    # N = All - C
    N = all_points_set - C
    
    # Convert to array
    points = []
    for pt in N:
        x = pt // q
        y = pt % q
        xa, xb = x // p, x % p
        ya, yb = y // p, y % p
        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
