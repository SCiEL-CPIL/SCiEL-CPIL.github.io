# EVOLVE-BLOCK-START
import numpy as np
import time
from collections import defaultdict

def nikodym_set() -> np.ndarray:
    """
    Construct a Nikodym set in F_49^2 using optimized local search.
    Start from known valid construction, then greedily add points to complement.
    """
    deadline = time.time() + 570
    
    p = 7
    q = p * p
    k = 3
    num_points = q * q
    num_dirs = q + 1
    
    # Precompute line groups
    line_groups = [None] * num_dirs
    for d in range(num_dirs):
        if d < q:
            ma = d % 7
            mb = d // 7
            groups = defaultdict(list)
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            c_a = (ya - ma * xa - 3 * mb * xb) % 7
                            c_b = (yb - ma * xb - mb * xa) % 7
                            pt = xa * 343 + xb * 49 + ya * 7 + yb
                            groups[(c_a, c_b)].append(pt)
            line_groups[d] = [np.array(v, dtype=np.int32) for v in groups.values()]
        else:
            groups = defaultdict(list)
            for xa in range(p):
                for xb in range(p):
                    for ya in range(p):
                        for yb in range(p):
                            pt = xa * 343 + xb * 49 + ya * 7 + yb
                            groups[(xa, xb)].append(pt)
            line_groups[d] = [np.array(v, dtype=np.int32) for v in groups.values()]
    
    # Precompute other_pts[d, P] = array of 48 other points on line through P in direction d
    other_pts = np.zeros((num_dirs, num_points, q - 1), dtype=np.int32)
    for d in range(num_dirs):
        for pts in line_groups[d]:
            for i in range(len(pts)):
                P = pts[i]
                others = np.concatenate([pts[:i], pts[i+1:]])
                other_pts[d, P] = others
    
    # Build initial S
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add(((u * u) % p, v))
    
    in_S = np.zeros(num_points, dtype=bool)
    for ya in range(p):
        for yb in range(p):
            rhs = (-ya * ya + k * yb * yb) % p
            xa = (rhs * 4) % p
            for xb in range(p):
                if (ya, yb) not in y_special:
                    pt = xa * 343 + xb * 49 + ya * 7 + yb
                    in_S[pt] = True
    
    # Compute safe_dir and safe_count
    safe_dir = np.ones((num_dirs, num_points), dtype=bool)
    for d in range(num_dirs):
        for P in range(num_points):
            if np.any(in_S[other_pts[d, P]]):
                safe_dir[d, P] = False
    
    safe_count = safe_dir.sum(axis=0).astype(np.int32)
    
    # Local search
    import random
    rng = random.Random(42)
    points_in_N = np.array([pt for pt in range(num_points) if not in_S[pt]], dtype=np.int32)
    best_S_size = int(in_S.sum())
    best_in_S = in_S.copy()
    
    max_restarts = 300
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
        
        in_S_local = in_S.copy()
        safe_dir_local = safe_dir.copy()
        safe_count_local = safe_count.copy()
        
        order = points_in_N.copy()
        rng.shuffle(order)
        
        for P in order:
            if time.time() > deadline:
                break
            if in_S_local[P]:
                continue
            
            # Vectorized check: for each d, check if any Q on line(d,P) has safe_dir and safe_count<=1
            valid = True
            for d in range(num_dirs):
                others = other_pts[d, P]
                sm = safe_dir_local[d, others]
                if np.any(sm):
                    if np.any(safe_count_local[others[sm]] <= 1):
                        valid = False
                        break
            
            if valid:
                in_S_local[P] = True
                for d in range(num_dirs):
                    others = other_pts[d, P]
                    sm = safe_dir_local[d, others]
                    if np.any(sm):
                        affected = others[sm]
                        safe_dir_local[d, affected] = False
                        safe_count_local[affected] -= 1
        
        S_size = int(in_S_local.sum())
        if S_size > best_S_size:
            best_S_size = S_size
            best_in_S = in_S_local.copy()
    
    points = []
    for pt in range(num_points):
        if not best_in_S[pt]:
            xa = pt // 343
            xb = (pt // 49) % 7
            ya = (pt // 7) % 7
            yb = pt % 7
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)

# EVOLVE-BLOCK-END
