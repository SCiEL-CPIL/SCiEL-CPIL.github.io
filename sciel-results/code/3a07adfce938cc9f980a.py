import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 (F_49 = F_7[alpha], alpha^2 = 3).
    Returns a small Nikodym set N, maximizing the complement size.
    """
    start_time = time.time()
    deadline = start_time + 570.0
    
    p = 7
    k = 3
    
    all_points = [(xa, xb, ya, yb) 
                  for xa in range(p) for xb in range(p) 
                  for ya in range(p) for yb in range(p)]
    
    # Directions: (1, m) for m in F_49, plus vertical (0,1)
    directions = [(ma, mb) for ma in range(p) for mb in range(p)]
    directions.append(None)  # vertical
    n_dirs = len(directions)
    
    # Precompute line assignments
    line_assignments = []
    for d in directions:
        la = {}
        for pt in all_points:
            xa, xb, ya, yb = pt
            if d is None:
                lid = (xa, xb)
            else:
                ma, mb = d
                c0 = (ya - xa*ma - xb*mb*k) % p
                c1 = (yb - xa*mb - xb*ma) % p
                lid = (c0, c1)
            la[pt] = lid
        line_assignments.append(la)
    
    # Build line -> points mapping
    dir_lines = []
    for la in line_assignments:
        lines = {}
        for pt, lid in la.items():
            if lid not in lines:
                lines[lid] = set()
            lines[lid].add(pt)
        dir_lines.append(lines)
    
    # Baseline S
    squares = set((u * u) % p for u in range(p))
    inv2 = pow(2, p-2, p)
    
    S = set()
    for ya in range(p):
        if ya in squares:
            continue
        for yb in range(p):
            xa = (inv2 * (-ya*ya + yb*yb*k)) % p
            for xb in range(p):
                S.add((xa, xb, ya, yb))
    
    # Compute safe_dirs: for each point, directions where it's the only S-point on its line
    safe_dirs = {pt: set() for pt in all_points}
    for d_idx in range(n_dirs):
        for lid, line_pts in dir_lines[d_idx].items():
            s_count = sum(1 for pt in line_pts if pt in S)
            if s_count <= 1:
                for pt in line_pts:
                    safe_dirs[pt].add(d_idx)
    
    # Verify baseline
    for pt in all_points:
        assert safe_dirs[pt], f"Baseline invalid at {pt}"
    
    # Greedy expansion
    candidates = [pt for pt in all_points if pt not in S]
    random.seed(42)
    random.shuffle(candidates)
    
    for x in candidates:
        if time.time() > deadline - 20:
            break
        
        # Determine if adding x is safe
        # For each direction, check the line containing x
        # If line had exactly 1 S-point, that point (and others) lose this direction
        # Check if any point would lose its last safe direction
        
        can_add = True
        for d_idx in range(n_dirs):
            la = line_assignments[d_idx]
            lid = la[x]
            line_pts = dir_lines[d_idx][lid]
            
            s_pts = [pt for pt in line_pts if pt in S]
            if len(s_pts) == 1:
                # Adding x makes this line have 2 S-points
                # All points on this line lose direction d_idx
                for pt in line_pts:
                    if d_idx in safe_dirs[pt] and len(safe_dirs[pt]) == 1:
                        can_add = False
                        break
            if not can_add:
                break
        
        if can_add:
            S.add(x)
            # Update safe_dirs
            for d_idx in range(n_dirs):
                la = line_assignments[d_idx]
                lid = la[x]
                line_pts = dir_lines[d_idx][lid]
                s_count = sum(1 for pt in line_pts if pt in S)
                if s_count <= 1:
                    for pt in line_pts:
                        safe_dirs[pt].add(d_idx)
                else:
                    for pt in line_pts:
                        safe_dirs[pt].discard(d_idx)
    
    # Build N
    points = []
    for pt in all_points:
        if pt not in S:
            xa, xb, ya, yb = pt
            points.append([[xa, xb], [ya, yb]])
    
    if not points:
        return np.array([[[0, 0], [0, 0]]], dtype=np.int64)
    
    return np.array(points, dtype=np.int64)
