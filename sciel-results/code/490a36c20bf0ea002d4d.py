import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3  # alpha^2 = 3
    deadline = time.time() + 600  # 600s limit, use 550s for safety
    
    # Field ops using integers 0..48 (a + b*alpha)
    def to_int(a, b):
        return a + b * p
    
    def from_int(n):
        return (n % p, n // p)
    
    def add_f(a, b):
        return (a + b) % (p * p)
    
    def sub_f(a, b):
        return (a - b) % (p * p)
    
    def mul_f(a, b):
        a0, a1 = from_int(a)
        b0, b1 = from_int(b)
        r0 = (a0 * b0 + a1 * b1 * k) % p
        r1 = (a0 * b1 + a1 * b0) % p
        return to_int(r0, r1)
    
    def inv_f(a):
        a0, a1 = from_int(a)
        denom = (a0 * a0 - a1 * a1 * k) % p
        if denom == 0:
            return 0
        d_inv = pow(denom, p - 2, p)
        return mul_f(to_int(a0, (-a1) % p), d_inv)
    
    # Precompute field tables
    add_table = np.zeros((p*p, p*p), dtype=np.int32)
    mul_table = np.zeros((p*p, p*p), dtype=np.int32)
    for a in range(p*p):
        for b in range(p*p):
            add_table[a, b] = add_f(a, b)
            mul_table[a, b] = mul_f(a, b)
    
    # Points in F_49^2: index = x * 49 + y where x, y in 0..48
    # Total 2401 points
    n_points = p * p * p * p
    
    # Construct Unital U: 2*x + y^2 = 0 in F_49
    U = np.zeros(n_points, dtype=bool)
    for x in range(p*p):
        for y in range(p*p):
            # 2*x + y^2 = 0
            val = (2 * x + mul_table[y, y]) % (p*p)
            if val == 0:
                U[x * p*p + y] = True
    
    # All lines: vertical (x = c) and slanted (y = m*x + b)
    # Store as list of arrays of point indices
    lines = []
    
    # Vertical lines: x = c, 49 points each
    for c in range(p*p):
        pts = np.array([c * p*p + y for y in range(p*p)], dtype=np.int32)
        lines.append(pts)
    
    # Slanted lines: y = m*x + b
    for m in range(p*p):
        for b in range(p*p):
            pts = np.array([x * p*p + (mul_table[x, m] + b) % (p*p) for x in range(p*p)], dtype=np.int32)
            lines.append(pts)
    
    n_lines = len(lines)
    
    # Precompute line membership: for each line, which points it contains
    # And for each point, which lines contain it
    point_lines = [set() for _ in range(n_points)]
    for li, L in enumerate(lines):
        for pt in L:
            point_lines[pt].add(li)
    
    # Outside U
    outside_U = np.where(~U)[0]
    outside_set = set(outside_U.tolist())
    
    # Greedy set cover: cover all outside_U points with lines
    # When we pick a line, we must add its U-intersection to H (N)
    # We want to minimize |H|
    
    uncovered = set(outside_U.tolist())
    H = set()
    
    # Precompute for each line: intersection with U
    line_U_inter = []
    for L in lines:
        I = set(pt for pt in L if U[pt])
        line_U_inter.append(I)
    
    # Greedy: pick line that covers most uncovered points per unit cost in H
    while uncovered and time.time() < deadline - 10:
        best_line_idx = -1
        best_score = -1
        
        for i in range(n_lines):
            if not line_U_inter[i] and i >= 0:
                pass  # still check coverage
            
            actual_new_cov = 0
            # Count uncovered points on this line
            for pt in lines[i]:
                if pt in uncovered:
                    actual_new_cov += 1
            
            if actual_new_cov == 0:
                continue
            
            new_H_cost = len(line_U_inter[i] - H)
            score = actual_new_cov / (new_H_cost + 1)
            
            if score > best_score:
                best_score = score
                best_line_idx = i
        
        if best_line_idx == -1:
            break
        
        # Pick best line
        L = lines[best_line_idx]
        I_L = line_U_inter[best_line_idx]
        H.update(I_L)
        
        for pt in L:
            if pt in uncovered:
                uncovered.remove(pt)
    
    # C = U - H (complement points in U)
    C = set(pt for pt in range(n_points) if U[pt] and pt not in H)
    
    # N = All - C
    N = [pt for pt in range(n_points) if pt not in C]
    
    # Convert to array
    points = []
    for pt in N:
        x = pt // (p*p)
        y = pt % (p*p)
        xa, xb = from_int(x)
        ya, yb = from_int(y)
        points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
