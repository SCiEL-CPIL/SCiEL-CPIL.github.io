import numpy as np
import random
import time

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3
    q = 49
    N_pts = q * q
    N_dirs = 50
    
    # Precompute affected
    # affected_x[c] = list of x indices
    # affected_d[c] = list of d indices
    # Or just list of (x, d)
    
    # To speed up, let's precompute for each c, the list of flat indices in count
    # count_idx = x * 50 + d
    
    affected_flat = [[] for _ in range(N_pts)]
    
    for c_idx in range(N_pts):
        xa = c_idx // 343
        xb = (c_idx // 49) % 7
        ya = (c_idx // 7) % 7
        yb = c_idx % 7
        
        # d 0..48: m in F_49
        for ma in range(p):
            for mb in range(p):
                d_idx = ma * 7 + mb
                for t_xa in range(p):
                    for t_xb in range(p):
                        dx_a = (t_xa - xa) % p
                        dx_b = (t_xb - xb) % p
                        if dx_a == 0 and dx_b == 0:
                            continue
                        dy_a = (ma * dx_a - k * mb * dx_b) % p
                        dy_b = (ma * dx_b + mb * dx_a) % p
                        ty_a = (ya + dy_a) % p
                        ty_b = (yb + dy_b) % p
                        x_idx = t_xa * 343 + t_xb * 49 + ty_a * 7 + ty_b
                        affected_flat[c_idx].append(x_idx * 50 + d_idx)
        
        # d 49: vertical
        d_idx = 49
        for t_ya in range(p):
            for t_yb in range(p):
                if t_ya == ya and t_yb == yb:
                    continue
                x_idx = xa * 343 + xb * 49 + t_ya * 7 + t_yb
                affected_flat[c_idx].append(x_idx * 50 + d_idx)
                
    def construct():
        C = np.zeros(N_pts, dtype=bool)
        count = np.zeros(N_pts * 50, dtype=np.int8) # max 48, int8 is enough
        clean_count = np.full(N_pts, 50, dtype=np.int8)
        
        available = list(range(N_pts))
        random.shuffle(available)
        
        added = 0
        no_improve = 0
        max_no_improve = 200
        
        while available:
            added_this_step = False
            # Try first few random points
            # We can limit the number of trials to save time
            for i in range(min(50, len(available))):
                c_idx = available[i]
                if C[c_idx]:
                    continue
                
                if clean_count[c_idx] == 0:
                    continue
                
                safe = True
                for flat_idx in affected_flat[c_idx]:
                    if count[flat_idx] == 0:
                        x_idx = flat_idx // 50
                        if clean_count[x_idx] == 1:
                            safe = False
                            break
                
                if safe:
                    C[c_idx] = True
                    added += 1
                    available.pop(i)
                    added_this_step = True
                    
                    for flat_idx in affected_flat[c_idx]:
                        count[flat_idx] += 1
                        if count[flat_idx] == 1:
                            x_idx = flat_idx // 50
                            clean_count[x_idx] -= 1
                    break
            
            if not added_this_step:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
            else:
                no_improve = 0
                if len(available) > 50:
                    random.shuffle(available)
                    
        return C

    best_C = None
    best_size = 0
    start_time = time.time()
    timeout = 500
    
    while time.time() - start_time < timeout:
        C = construct()
        size = np.sum(C)
        if size > best_size:
            best_size = size
            best_C = C
            
    # Construct N
    N_mask = ~best_C
    points = []
    for i in range(N_pts):
        if N_mask[i]:
            xa = i // 343
            xb = (i // 49) % 7
            ya = (i // 7) % 7
            yb = i % 7
            points.append([[xa, xb], [ya, yb]])
            
    return np.array(points, dtype=np.int64)
