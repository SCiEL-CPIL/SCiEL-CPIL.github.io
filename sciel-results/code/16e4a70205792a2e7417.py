import numpy as np
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p  # 49
    k = 3  # alpha^2 = 3
    
    def add(e1, e2):
        return ((e1[0] + e2[0]) % p, (e1[1] + e2[1]) % p)
    
    def mul(e1, e2):
        a1, b1 = e1
        a2, b2 = e2
        return ((a1*a2 + b1*b2*k) % p, (a1*b2 + b1*a2) % p)
    
    def neg(e):
        return ((-e[0]) % p, (-e[1]) % p)
    
    field_elements = [(a, b) for a in range(p) for b in range(p)]
    num_points = q * q  # 2401
    
    # Directions
    directions = []
    for m in field_elements:
        directions.append(((1, 0), m))
    directions.append(((0, 0), (1, 0)))
    num_dirs = len(directions)  # 50
    
    def idx_to_point(idx):
        y_idx = idx % q
        x_idx = idx // q
        xb = y_idx % p
        ya = y_idx // p
        xb = x_idx % p
        xa = x_idx // p
        return ((xa, xb), (ya, yb))
    
    all_points = [idx_to_point(i) for i in range(num_points)]
    
    # Precompute lines
    lines = [None] * num_dirs
    for d in range(num_dirs):
        vx, vy = directions[d]
        line_groups = {}
        for pt_idx in range(num_points):
            pt = all_points[pt_idx]
            if vx == (1, 0):
                m = vy
                b = add(pt[1], neg(mul(pt[0], m)))
                key = ('s', m, b)
            elif vx == (0, 0):
                key = ('v', pt[0])
            else:
                key = pt_idx
            if key not in line_groups:
                line_groups[key] = []
            line_groups[key].append(pt_idx)
        lines[d] = list(line_groups.values())
    
    # Precompute line_neighbors
    line_neighbors = [None] * num_dirs
    for d in range(num_dirs):
        line_neighbors[d] = [None] * num_points
        for line in lines[d]:
            for pt_idx in line:
                line_neighbors[d][pt_idx] = [pt2 for pt2 in line if pt2 != pt_idx]
    
    start_time = time.time()
    deadline = start_time + 580  # leave 20s margin
    
    best_S_count = 0
    best_in_S = None
    
    def run_greedy(order):
        c = np.zeros((num_points, num_dirs), dtype=np.int32)
        in_S = np.zeros(num_points, dtype=bool)
        covered_count = np.full(num_points, num_dirs, dtype=np.int32)
        
        for s in order:
            if time.time() > deadline:
                break
            in_S[s] = True
            affected = False
            for d in range(num_dirs):
                for x in line_neighbors[d][s]:
                    old_c = c[x, d]
                    c[x, d] = old_c + 1
                    if old_c == 0:
                        covered_count[x] -= 1
                        if covered_count[x] == 0:
                            affected = True
            
            if affected:
                in_S[s] = False
                for d in range(num_dirs):
                    for x in line_neighbors[d][s]:
                        old_c = c[x, d]
                        c[x, d] = old_c - 1
                        if old_c == 1:
                            covered_count[x] += 1
        
        return in_S.sum(), in_S
    
    # Try sequential order
    order = list(range(num_points))
    count, in_S = run_greedy(order)
    if count > best_S_count:
        best_S_count = count
        best_in_S = in_S.copy()
    
    # Try random orders
    np.random.seed(42)
    for trial in range(20):
        if time.time() > deadline - 30:
            break
        order = list(range(num_points))
        np.random.shuffle(order)
        count, in_S = run_greedy(order)
        if count > best_S_count:
            best_S_count = count
            best_in_S = in_S.copy()
    
    # Build N
    N_points = []
    for i in range(num_points):
        if not best_in_S[i]:
            pt = all_points[i]
            N_points.append([pt[0], pt[1]])
    
    return np.array(N_points, dtype=np.int64)
