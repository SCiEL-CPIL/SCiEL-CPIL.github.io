import numpy as np
import time

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    Returns a small Nikodym set N, maximizing the complement size.
    """
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    # Element (a,b) represents a + b*alpha
    # Multiplication: (a,b)*(c,d) = (ac + bd*k, ad + bc) mod p
    def mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
    
    # Directions: 49 of form (1, m) and 1 vertical (0, 1)
    # For direction (1, m) where m = (ma, mb):
    # Line through (x,y): {(x + t, y + t*m) : t in F_49}
    # Invariant: y - x*m = constant
    # (ya, yb) - (xa, xb)*(ma, mb) = (ya - xa*ma - xb*mb*k, yb - xa*mb - xb*ma) mod p
    
    # For vertical direction (0, 1):
    # Line through (x,y): {(x, y + t) : t in F_49}
    # Invariant: x = constant, i.e., (xa, xb) = constant
    
    # Precompute line IDs for each point and direction
    # directions: 0..48 are (1, m) with m = (i, j) for i,j in 0..6
    # direction 49 is vertical
    
    n_dirs = 50
    n_points = p * p * p * p  # 2401
    
    # line_id[dir][point] = line identifier (tuple of two ints)
    # For direction (1, m): line_id = (ya - xa*ma - xb*mb*k, yb - xa*mb - xb*ma) mod p
    # For vertical: line_id = (xa, xb)
    
    # Precompute for each direction, a mapping from line_id -> list of point indices
    # point index = xa * p^3 + xb * p^2 + ya * p + yb
    
    def point_to_idx(xa, xb, ya, yb):
        return xa * p**3 + xb * p**2 + ya * p + yb
    
    def idx_to_point(idx):
        yb = idx % p
        ya = (idx // p) % p
        xb = (idx // p**2) % p
        xa = (idx // p**3) % p
        return (xa, xb, ya, yb)
    
    # Precompute line assignments
    line_assignments = []  # list of dicts: dir_idx -> {line_id: [point_indices]}
    
    for d_idx in range(n_dirs):
        lines = {}
        for xa in range(p):
            for xb in range(p):
                for ya in range(p):
                    for yb in range(p):
                        idx = point_to_idx(xa, xb, ya, yb)
                        if d_idx < 49:
                            ma = d_idx // p
                            mb = d_idx % p
                            c0 = (ya - xa * ma - xb * mb * k) % p
                            c1 = (yb - xa * mb - xb * ma) % p
                            line_id = (c0, c1)
                        else:
                            line_id = (xa, xb)
                        
                        if line_id not in lines:
                            lines[line_id] = []
                        lines[line_id].append(idx)
        line_assignments.append(lines)
    
    # Build the baseline S: points on the parabola-like curve
    # For y_a not a square in F_7:
    # squares in F_7: {0, 1, 2, 4}
    # non-squares: {3, 5, 6}
    squares = {0, 1, 2, 4}
    inv2 = pow(2, p-2, p)
    
    S = set()
    for ya in range(p):
        if ya in squares:
            continue
        for yb in range(p):
            # xa = inv2 * (-ya^2 + yb^2 * k) mod p
            xa = (inv2 * (-ya*ya + yb*yb*k)) % p
            for xb in range(p):
                idx = point_to_idx(xa, xb, ya, yb)
                S.add(idx)
    
    baseline_size = len(S)
    
    # For each point, track which directions are "safe" (point is the only S-point on that line)
    safe_dirs = [set() for _ in range(n_points)]
    
    for d_idx in range(n_dirs):
        lines = line_assignments[d_idx]
        for line_id, pts in lines.items():
            s_count = 0
            for idx in pts:
                if idx in S:
                    s_count += 1
            if s_count <= 1:
                for idx in pts:
                    safe_dirs[idx].add(d_idx)
    
    # Verify all points have at least one safe direction
    for idx in range(n_points):
        if not safe_dirs[idx]:
            # Fallback: return just the complement of baseline
            pass
    
    # Greedy expansion: try to add points to S
    candidates = [idx for idx in range(n_points) if idx not in S]
    
    import random
    random.seed(42)
    random.shuffle(candidates)
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    added = 0
    for x_idx in candidates:
        if time.time() > deadline - 30:
            break
        
        # Check if we can add x_idx
        # Adding x_idx affects: for each direction, the line containing x_idx
        # If that line already has an S-point, those points may lose safety
        # We need to check if any point would lose its last safe direction
        
        can_add = True
        affected_pairs = []  # (point_idx, dir_idx)
        
        for d_idx in range(n_dirs):
            lines = line_assignments[d_idx]
            # Find the line containing x_idx
            xa, xb, ya, yb = idx_to_point(x_idx)
            if d_idx < 49:
                ma = d_idx // p
                mb = d_idx % p
                c0 = (ya - xa * ma - xb * mb * k) % p
                c1 = (yb - xa * mb - xb * ma) % p
                line_id = (c0, c1)
            else:
                line_id = (xa, xb)
            
            pts = lines[line_id]
            
            # Count S points on this line (excluding x_idx since not in S yet)
            s_count = 0
            for idx in pts:
                if idx in S:
                    s_count += 1
            
            if s_count >= 1:
                # Adding x_idx would make s_count + 1 >= 2
                # So all points on this line lose direction d_idx from safe_dirs
                for idx in pts:
                    if d_idx in safe_dirs[idx]:
                        if len(safe_dirs[idx]) == 1:
                            can_add = False
                            break
                        affected_pairs.append((idx, d_idx))
                if not can_add:
                    break
        
        if can_add:
            S.add(x_idx)
            added += 1
            
            # Update safe_dirs
            for d_idx in range(n_dirs):
                xa, xb, ya, yb = idx_to_point(x_idx)
                if d_idx < 49:
                    ma = d_idx // p
                    mb = d_idx % p
                    c0 = (ya - xa * ma - xb * mb * k) % p
                    c1 = (yb - xa * mb - xb * ma) % p
                    line_id = (c0, c1)
                else:
                    line_id = (xa, xb)
                
                lines = line_assignments[d_idx]
                pts = lines[line_id]
                
                s_count = 0
                for idx in pts:
                    if idx in S:
                        s_count += 1
                
                if s_count <= 1:
                    for idx in pts:
                        safe_dirs[idx].add(d_idx)
                else:
                    for idx in pts:
                        safe_dirs[idx].discard(d_idx)
    
    final_size = len(S)
    
    # Build N = complement of S
    points = []
    for idx in range(n_points):
        if idx not in S:
            xa, xb, ya, yb = idx_to_point(idx)
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
    
    return np.array(points, dtype=np.int64)
