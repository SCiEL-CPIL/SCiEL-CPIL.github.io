import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    elems = [(a, b) for a in range(p) for b in range(p)]
    idx_of_elem = {e: i for i, e in enumerate(elems)}
    
    def add_f(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
        
    def mul_f(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1*a2 + 3*b1*b2) % p, (a1*b2 + a2*b1) % p)
        
    add_table = np.zeros((49, 49), dtype=np.int64)
    mul_table = np.zeros((49, 49), dtype=np.int64)
    for i in range(49):
        for j in range(49):
            add_table[i, j] = idx_of_elem[add_f(elems[i], elems[j])]
            mul_table[i, j] = idx_of_elem[mul_f(elems[i], elems[j])]
            
    def get_idx(xa, xb, ya, yb):
        return ((xa * 7 + xb) * 7 + ya) * 7 + yb
        
    def get_coords(idx):
        yb = idx % 7
        ya = (idx // 7) % 7
        xb = (idx // 49) % 7
        xa = (idx // 343) % 7
        return xa, xb, ya, yb
        
    line_points = np.full((2401, 50, 48), -1, dtype=np.int64)
    
    # 1. Vertical lines
    for xa in range(p):
        for xb in range(p):
            pts = []
            for ya in range(p):
                for yb in range(p):
                    pts.append(get_idx(xa, xb, ya, yb))
            for i, pt in enumerate(pts):
                for j, other in enumerate(pts):
                    if i != j:
                        line_points[pt, 0, j] = other
                        
    # 2. Slanted lines
    for m_idx in range(49):
        m = elems[m_idx]
        dir_idx = 1 + m_idx
        for b_idx in range(49):
            b = elems[b_idx]
            pts = []
            for x_idx in range(49):
                x = elems[x_idx]
                y = add_f(mul_f(m, x), b)
                ya, yb = y
                xa, xb = x
                pts.append(get_idx(xa, xb, ya, yb))
            for i, pt in enumerate(pts):
                for j, other in enumerate(pts):
                    if i != j:
                        line_points[pt, dir_idx, j] = other
                        
    best_S = set()
    
    start_time = time.time()
    timeout = 550 # Leave margin for final construction
    
    for seed in range(10000):
        if time.time() - start_time > timeout:
            break
            
        np.random.seed(seed)
        order = np.random.permutation(2401)
        
        is_valid = np.ones((2401, 50), dtype=bool)
        valid_count = np.full(2401, 50, dtype=np.int64)
        S = set()
        
        for y in order:
            if valid_count[y] == 0:
                continue
                
            safe = True
            for v in range(50):
                neighbors = line_points[y, v]
                valid_flags = is_valid[neighbors, v]
                if np.any(valid_flags):
                    counts = valid_count[neighbors[valid_flags]]
                    if np.any(counts == 1):
                        safe = False
                        break
                        
            if safe:
                S.add(y)
                for v in range(50):
                    neighbors = line_points[y, v]
                    valid_flags = is_valid[neighbors, v]
                    if np.any(valid_flags):
                        is_valid[neighbors[valid_flags], v] = False
                        valid_count[neighbors[valid_flags]] -= 1
                        
        if len(S) > len(best_S):
            best_S = S.copy()
            
    points = []
    for i in range(2401):
        if i not in best_S:
            xa, xb, ya, yb = get_coords(i)
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
