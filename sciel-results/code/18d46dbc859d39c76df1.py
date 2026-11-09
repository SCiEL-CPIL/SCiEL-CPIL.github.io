import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    
    def field_add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def field_mul(a, b):
        return ((a[0]*b[0] + a[1]*b[1]*k) % p, (a[0]*b[1] + a[1]*b[0]) % p)
    
    # Enumerate U
    U_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    if (2*xa + ya*ya - yb*yb*k) % p == 0:
                        U_points.append(((xa, xb), (ya, yb)))
    
    n_U = len(U_points)
    
    # All points
    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
    
    n_all = len(all_points)  # 2401
    
    # Point to index mapping
    pt_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    # U point indices
    U_indices = [pt_to_idx[pt] for pt in U_points]
    is_U = [False] * n_all
    for i in U_indices:
        is_U[i] = True
    
    # Directions
    slopes = [(ma, mb) for ma in range(p) for mb in range(p)]
    n_dirs = 50
    
    # Precompute lines: lines[pt_idx][d] = set of point indices on the line
    # This is 2401 * 50 * 49 = ~5.9 million entries, too much memory
    # Instead, precompute only for U points and for checking condition 2
    
    # For condition 1: for each U point and each direction, which U points are on the line
    u_on_line = [None] * n_U
    for i in range(n_U):
        u_on_line[i] = [set() for _ in range(n_dirs)]
        pt = U_points[i]
        for d in range(n_dirs):
            if d < 49:
                m = slopes[d]
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        new_x = field_add(pt[0], t)
                        mt = field_mul(m, t)
                        new_y = field_add(pt[1], mt)
                        other = (new_x, new_y)
                        if other in pt_to_idx:
                            j = pt_to_idx[other]
                            if is_U[j] and j != pt_to_idx[pt]:
                                u_on_line[i][d].add(j)
            else:
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        new_y = field_add(pt[1], t)
                        other = (pt[0], new_y)
                        if other in pt_to_idx:
                            j = pt_to_idx[other]
                            if is_U[j] and j != pt_to_idx[pt]:
                                u_on_line[i][d].add(j)
    
    # For condition 2: for each non-U point and each direction, which U points are on the line
    # Only needed for points not in S, but let's precompute for all non-U points
    # Actually, we need it for all points not in S, which includes U points not in S
    # Let's precompute for all points
    pt_U_on_line = [None] * n_all
    for i in range(n_all):
        pt_U_on_line[i] = [set() for _ in range(n_dirs)]
        pt = all_points[i]
        for d in range(n_dirs):
            if d < 49:
                m = slopes[d]
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        new_x = field_add(pt[0], t)
                        mt = field_mul(m, t)
                        new_y = field_add(pt[1], mt)
                        other = (new_x, new_y)
                        if other in pt_to_idx:
                            j = pt_to_idx[other]
                            if is_U[j]:
                                pt_U_on_line[i][d].add(j)
            else:
                for ta in range(p):
                    for tb in range(p):
                        t = (ta, tb)
                        new_y = field_add(pt[1], t)
                        other = (pt[0], new_y)
                        if other in pt_to_idx:
                            j = pt_to_idx[other]
                            if is_U[j]:
                                pt_U_on_line[i][d].add(j)
    
    # Now, S is a subset of U. Represent S as a set of U-point indices (0 to n_U-1)
    # S_set: set of U indices in S
    
    def check_validity(S_u_indices):
        """Check if S (given as set of U indices) is a valid Nikodym complement."""
        S_set = set(S_u_indices)
        
        # Condition 1: for each s in S, ∃ d with u_on_line[s][d] ∩ S_set = ∅
        for s in S_set:
            has_iso = False
            for d in range(n_dirs):
                if not u_on_line[s][d].intersection(S_set):
                    has_iso = True
                    break
            if not has_iso:
                return False
        
        # Condition 2: for each x not in S, ∃ d with pt_U_on_line[x][d] ∩ S_set = ∅
        # x not in S means: x is not a U point in S
        for x in range(n_all):
            if is_U[x] and (U_indices.index(x) in S_set):
                continue  # x is in S, skip
            # x is in N, need a clean direction
            has_clean = False
            for d in range(n_dirs):
                if not pt_U_on_line[x][d].intersection(S_set):
                    has_clean = True
                    break
            if not has_clean:
                return False
        
        return True
    
    # Start with baseline S
    # Baseline: S = U \ B where B = U ∩ {y in Y}
    y_special = set()
    for u in range(p):
        for v in range(p):
            y_special.add((u*u % p, v))
    
    baseline_S = set()
    for i in range(n_U):
        pt = U_points[i]
        ya, yb = pt[1]
        if (ya, yb) not in y_special:
            baseline_S.add(i)
    
    # Try to expand S
    S = set(baseline_S)
    
    # Greedy: try to add each U point not in S
    for i in range(n_U):
        if i in S:
            continue
        S_new = S | {i}
        if check_validity(S_new):
            S = S_new
    
    # Build N
    S_points = set(U_points[i] for i in S)
    points = []
    for pt in all_points:
        if pt not in S_points:
            xa, xb = pt[0]
            ya, yb = pt[1]
            points.append([[xa, xb], [ya, yb]])
    
    return np.array(points, dtype=np.int64)
