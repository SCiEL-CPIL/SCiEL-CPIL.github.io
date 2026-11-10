import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    
    # Field F_49 = F_7[alpha], alpha^2 = 3
    # k = 3 is a non-residue mod 7?
    # 1^2=1, 2^2=4, 3^2=2, 4^2=2, 5^2=4, 6^2=1.
    # Squares: {0, 1, 2, 4}.
    # Non-squares: {3, 5, 6}.
    # Seed used k=3.
    k = 3
    
    # Map (a, b) to index 0..2400
    def to_idx(a, b):
        return a * p + b
        
    def to_coords(idx):
        a = idx // p
        b = idx % p
        return a, b
        
    # Precompute field operations
    # add, mul tables for F_49
    # Element is (a, b) representing a + b*alpha
    # (a1, b1) + (a2, b2) = ((a1+a2)%p, (b1+b2)%p)
    # (a1, b1) * (a2, b2) = (a1*a2 + b1*b2*k, a1*b2 + a2*b1) mod p
    
    # Generate all points
    points = []
    for a in range(p):
        for b in range(p):
            for c in range(p):
                for d in range(p):
                    points.append((a, b, c, d))
                    
    # Point index: (xa, xb, ya, yb) -> idx
    # idx = ((xa * p + xb) * p + ya) * p + yb
    def point_idx(xa, xb, ya, yb):
        return ((xa * p + xb) * p + ya) * p + yb
        
    def point_coords(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // (p*p)) % p
        xa = (idx // (p*p*p)) % p
        return xa, xb, ya, yb

    # Define U
    in_U = np.zeros(q*q, dtype=bool)
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2 * xa + ya * ya - yb * yb * k) % p == 0:
                        in_U[point_idx(xa, xb, ya, yb)] = True
                        
    S = in_U.copy()
    
    # Precompute lines
    # Line 0..48: Vertical x = c_a + c_b alpha
    # Line 49..: y = m x + c
    # m, c in F_49
    
    lines_points = []
    point_lines = [[] for _ in range(q*q)]
    
    line_id = 0
    # Vertical lines
    for ca in range(p):
        for cb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    idx = point_idx(ca, cb, ya, yb)
                    pts.append(idx)
                    point_lines[idx].append(line_id)
            lines_points.append(pts)
            line_id += 1
            
    # Non-vertical lines
    # m = (ma, mb), c = (ca, cb)
    for ma in range(p):
        for mb in range(p):
            for ca in range(p):
                for cb in range(p):
                    pts = []
                    for xa in range(p):
                        for xb in range(p):
                            # y = m*x + c
                            # x = xa + xb*alpha
                            # m*x = (ma + mb*alpha)(xa + xb*alpha) = ma*xa + mb*xb*k + (ma*xb + mb*xa)*alpha
                            ya = (ma * xa + mb * xb * k + ca) % p
                            yb = (ma * xb + mb * xa + cb) % p
                            idx = point_idx(xa, xb, ya, yb)
                            pts.append(idx)
                            point_lines[idx].append(line_id)
                    lines_points.append(pts)
                    line_id += 1
                    
    # Pruning loop
    while True:
        # Count points in S on each line
        line_counts = np.zeros(line_id, dtype=int)
        for lid, pts in enumerate(lines_points):
            count = 0
            for idx in pts:
                if S[idx]:
                    count += 1
            line_counts[lid] = count
            
        # Check which points in S are valid
        to_remove = []
        for idx in range(q*q):
            if not S[idx]:
                continue
            # Check if any line through idx has count 1
            is_loner = False
            for lid in point_lines[idx]:
                if line_counts[lid] == 1:
                    is_loner = True
                    break
            if not is_loner:
                to_remove.append(idx)
                
        if not to_remove:
            break
            
        for idx in to_remove:
            S[idx] = False
            
    # Construct N
    N_indices = []
    for idx in range(q*q):
        if not S[idx]:
            N_indices.append(idx)
            
    # Convert to array
    result = np.zeros((len(N_indices), 2, 2), dtype=np.int64)
    for i, idx in enumerate(N_indices):
        xa, xb, ya, yb = point_coords(idx)
        result[i, 0, 0] = xa
        result[i, 0, 1] = xb
        result[i, 1, 0] = ya
        result[i, 1, 1] = yb
        
    return result
