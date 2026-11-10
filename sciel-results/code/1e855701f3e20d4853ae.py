import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    
    # Field elements are tuples (a, b)
    # Generate all field elements
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    
    def add_f(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
    
    def mul_f(a, b):
        return ((a[0]*b[0] + 3*a[1]*b[1]) % p, (a[0]*b[1] + a[1]*b[0]) % p)
        
    def sub_f(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)

    # Generate all points in F_49^2
    # Point is (x, y) where x, y in field_elements
    points = []
    point_to_idx = {}
    for x in field_elements:
        for y in field_elements:
            pt = (x, y)
            idx = len(points)
            points.append(pt)
            point_to_idx[pt] = idx
            
    n_points = len(points)
    
    # Precompute lines
    line_canonical_to_id = {}
    lines = []
    point_lines = [[] for _ in range(n_points)]
    line_id = 0
    
    # Vertical lines: x = const
    for x in field_elements:
        key = ('V', x)
        if key not in line_canonical_to_id:
            line_canonical_to_id[key] = line_id
            lines.append([])
            for pt in points:
                if pt[0] == x:
                    idx = point_to_idx[pt]
                    lines[-1].append(idx)
                    point_lines[idx].append(line_id)
            line_id += 1
            
    # Non-vertical lines: y = m x + c
    for m in field_elements:
        seen = set()
        for p0 in points:
            x0, y0 = p0
            mx0 = mul_f(m, x0)
            c = sub_f(y0, mx0)
            key = (m, c)
            if key in seen:
                continue
            seen.add(key)
            
            if key not in line_canonical_to_id:
                line_canonical_to_id[key] = line_id
                lines.append([])
                for pt in points:
                    x, y = pt
                    mx = mul_f(m, x)
                    y_calc = add_f(mx, c)
                    if y == y_calc:
                        idx = point_to_idx[pt]
                        lines[-1].append(idx)
                        point_lines[idx].append(line_id)
                line_id += 1
                
    n_lines = line_id
    
    start_time = time.time()
    deadline = start_time + 500
    
    best_S_size = 0
    best_in_S = None
    
    while time.time() < deadline:
        in_S = [False] * n_points
        S_count = [0] * n_lines
        free_count = [50] * n_points
        S_size = 0
        
        order = list(range(n_points))
        random.shuffle(order)
        
        for y in order:
            valid_add = True
            changes = []
            
            for L in point_lines[y]:
                old_count = S_count[L]
                for x in lines[L]:
                    if x == y:
                        continue
                    was_free = False
                    if in_S[x]:
                        if old_count == 1:
                            was_free = True
                    else:
                        if old_count == 0:
                            was_free = True
                            
                    if was_free:
                        free_count[x] -= 1
                        if free_count[x] == 0:
                            valid_add = False
                        changes.append(x)
                        
                S_count[L] = old_count + 1
                
            if valid_add:
                in_S[y] = True
                S_size += 1
            else:
                for L in point_lines[y]:
                    S_count[L] -= 1
                for x in changes:
                    free_count[x] += 1
                    
        if S_size > best_S_size:
            best_S_size = S_size
            best_in_S = in_S[:]
            
    N_points = []
    for i in range(n_points):
        if not best_in_S[i]:
            x, y = points[i]
            N_points.append([[x[0], x[1]], [y[0], y[1]]])
            
    return np.array(N_points, dtype=np.int64)
