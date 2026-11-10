import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    q = p * p
    k = 3
    
    def f_to_int(a, b):
        return a * p + b
        
    def int_to_f(z):
        return (z // p, z % p)
        
    def f_add(a, b):
        aa, ab = int_to_f(a)
        ba, bb = int_to_f(b)
        return f_to_int((aa+ba)%p, (ab+bb)%p)
        
    def f_mul(a, b):
        aa, ab = int_to_f(a)
        ba, bb = int_to_f(b)
        res_a = (aa*ba + ab*bb*k) % p
        res_b = (aa*bb + ab*ba) % p
        return f_to_int(res_a, res_b)

    # Precompute lines
    lines = []
    point_to_lines = [[] for _ in range(q*q)]
    
    def add_line(pts):
        line_idx = len(lines)
        lines.append(set(pts))
        for pt in pts:
            point_to_lines[pt].append(line_idx)
            
    for c in range(q):
        pts = [c * q + y for y in range(q)]
        add_line(pts)
        
    for m in range(q):
        for c in range(q):
            pts = []
            for x in range(q):
                mx = f_mul(m, x)
                y = f_add(mx, c)
                pts.append(x * q + y)
            add_line(pts)
            
    def solve(seed):
        random.seed(seed)
        in_C = [True] * (q*q)
        count_C = [len(l) for l in lines]
        
        def update_valid_counts():
            valid_lines_count = [0] * (q*q)
            for pt in range(q*q):
                vc = 0
                for l_idx in point_to_lines[pt]:
                    if in_C[pt]:
                        if count_C[l_idx] == 1:
                            vc += 1
                    else:
                        if count_C[l_idx] == 0:
                            vc += 1
                valid_lines_count[pt] = vc
            return valid_lines_count

        valid_lines_count = update_valid_counts()
        
        # Greedy Removal
        while True:
            uncovered = [pt for pt in range(q*q) if valid_lines_count[pt] == 0]
            if not uncovered:
                break
            
            pt = random.choice(uncovered)
            best_l = -1
            best_cost = q + 1
            
            if in_C[pt]:
                for l_idx in point_to_lines[pt]:
                    cost = count_C[l_idx] - 1
                    if cost < best_cost:
                        best_cost = cost
                        best_l = l_idx
            else:
                for l_idx in point_to_lines[pt]:
                    cost = count_C[l_idx]
                    if cost < best_cost:
                        best_cost = cost
                        best_l = l_idx
                        
            if best_l == -1:
                break
                
            if in_C[pt]:
                to_remove = [p for p in lines[best_l] if in_C[p] and p != pt]
            else:
                to_remove = [p for p in lines[best_l] if in_C[p]]
                
            for p in to_remove:
                in_C[p] = False
                for l_idx in point_to_lines[p]:
                    count_C[l_idx] -= 1
                    
            valid_lines_count = update_valid_counts()
            
        # Local Search
        improved = True
        iterations = 0
        while improved and iterations < 2000:
            improved = False
            iterations += 1
            candidates = [p for p in range(q*q) if not in_C[p]]
            if not candidates:
                break
            random.shuffle(candidates)
            
            for z in candidates:
                # Simulate add z
                affected_lines = point_to_lines[z]
                old_counts = [count_C[l_idx] for l_idx in affected_lines]
                
                in_C[z] = True
                for i, l_idx in enumerate(affected_lines):
                    count_C[l_idx] += 1
                    
                # Check validity
                # We can just recalculate valid_lines_count for all points?
                # Or check if any point has 0 valid lines.
                # Let's recalculate all to be safe.
                new_valid = update_valid_counts()
                if all(v > 0 for v in new_valid):
                    improved = True
                    valid_lines_count = new_valid
                    break
                else:
                    # Revert
                    in_C[z] = False
                    for i, l_idx in enumerate(affected_lines):
                        count_C[l_idx] = old_counts[i]
                    
        # Count N
        n_size = sum(1 for p in range(q*q) if not in_C[p])
        return n_size, in_C

    start_time = time.time()
    deadline = 600.0
    
    best_size = q*q
    best_in_C = None
    
    for seed in range(10):
        if time.time() > deadline - 30:
            break
        n_size, in_C = solve(seed)
        if n_size < best_size:
            best_size = n_size
            best_in_C = in_C
            
    n_points = []
    for p in range(q*q):
        if not best_in_C[p]:
            x = p // q
            y = p % q
            xa, xb = int_to_f(x)
            ya, yb = int_to_f(y)
            n_points.append([[xa, xb], [ya, yb]])
            
    return np.array(n_points, dtype=np.int64)
