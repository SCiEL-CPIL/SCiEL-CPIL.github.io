import numpy as np
import random

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3
    
    def add(a, b):
        return ((a[0] + b[0]) % p, (a[1] + b[1]) % p)
        
    def sub(a, b):
        return ((a[0] - b[0]) % p, (a[1] - b[1]) % p)
        
    def mul(a, b):
        a0, a1 = a
        b0, b1 = b
        return ((a0*b0 + a1*b1*k) % p, (a0*b1 + a1*b0) % p)

    all_points = []
    for xa in range(p):
        for xb in range(p):
            for ya in range(p):
                for yb in range(p):
                    all_points.append(((xa, xb), (ya, yb)))
                    
    point_to_idx = {pt: i for i, pt in enumerate(all_points)}
    
    F_q_elems = [(a, b) for a in range(p) for b in range(p)]
    
    # Generate all lines
    lines = []
    for c in F_q_elems:
        pts = set()
        for t in F_q_elems:
            pt = (c, t)
            pts.add(point_to_idx[pt])
        lines.append(pts)
        
    for m in F_q_elems:
        for b in F_q_elems:
            pts = set()
            for t in F_q_elems:
                y = add(mul(m, t), b)
                pt = (t, y)
                pts.add(point_to_idx[pt])
            lines.append(pts)
            
    best_N = None
    best_size = q*q
    
    for num_lines in range(1, 15):
        for _ in range(50):
            idxs = random.sample(range(len(lines)), num_lines)
            N_set = set()
            for idx in idxs:
                N_set.update(lines[idx])
                
            C_set = set(range(q*q)) - N_set
            
            valid = True
            for c_idx in C_set:
                c = all_points[c_idx]
                found_line = False
                
                # Check vertical
                ok = True
                for t in F_q_elems:
                    if t == c[1]: continue
                    pt = (c[0], t)
                    if point_to_idx[pt] not in N_set:
                        ok = False
                        break
                if ok:
                    found_line = True
                else:
                    for m in F_q_elems:
                        b = sub(c[1], mul(m, c[0]))
                        ok = True
                        for t in F_q_elems:
                            if t == c[0]: continue
                            y = add(mul(m, t), b)
                            pt = (t, y)
                            if point_to_idx[pt] not in N_set:
                                ok = False
                                break
                        if ok:
                            found_line = True
                            break
                if not found_line:
                    valid = False
                    break
            if valid:
                if len(N_set) < best_size:
                    best_size = len(N_set)
                    best_N = N_set
        
    if best_N is None:
        y_special = set()
        for u in range(p):
            for v in range(p):
                y_special.add(((u * u) % p, v))
                
        C_indices = set()
        for i, pt in enumerate(all_points):
            (xa, xb), (ya, yb) = pt
            in_unital = (2 * xa + ya * ya - yb * yb * k) % p == 0
            in_y_special = (ya, yb) in y_special
            if in_unital and not in_y_special:
                C_indices.add(i)
                
        points_in_N = []
        for i, pt in enumerate(all_points):
            if i not in C_indices:
                (xa, xb), (ya, yb) = pt
                points_in_N.append([[xa, xb], [ya, yb]])
        return np.array(points_in_N, dtype=np.int64)
        
    points_in_N = []
    for i, pt in enumerate(all_points):
        if i not in best_N:
            (xa, xb), (ya, yb) = pt
            points_in_N.append([[xa, xb], [ya, yb]])
            
    return np.array(points_in_N, dtype=np.int64)
