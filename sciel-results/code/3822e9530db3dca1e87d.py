import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_11():
    start_time = time.time()
    timeout = 600
    
    a = np.sqrt(3) / 2.0
    n_hex = 11
    min_side = np.sqrt(11)
    
    def is_valid(S, pos):
        if not np.all(np.isfinite(pos)) or not np.isfinite(S):
            return False
        if S < min_side:
            return False
            
        angles = np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        # Containment
        proj = pos @ normals.T
        max_proj = np.max(np.abs(proj), axis=1)
        limit = (S - 1) * a
        if np.any(max_proj > limit + 1e-6):
            return False
            
        # Overlap
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                c1 = pos[i]
                c2 = pos[j]
                p1 = c1 @ normals.T
                p2 = c2 @ normals.T
                low = np.maximum(p1 - a, p2 - a)
                high = np.minimum(p1 + a, p2 + a)
                gaps = high - low
                if np.max(gaps) < -1e-6:
                    return False
        return True

    def objective(v):
        if time.time() - start_time > timeout - 10:
            return 1e10
        S = v[0]
        if S < min_side:
            return 1e10
        pos = np.array(v[1:]).reshape(-1, 2)
        
        angles = np.arange(6) * np.pi / 3
        normals = np.column_stack((np.cos(angles), np.sin(angles)))
        
        proj = pos @ normals.T
        max_proj = np.max(np.abs(proj), axis=1)
        limit = (S - 1) * a
        viol_contain = np.sum(np.maximum(0, max_proj - limit))
        
        overlap_pen = 0.0
        for i in range(n_hex):
            for j in range(i+1, n_hex):
                c1 = pos[i]
                c2 = pos[j]
                p1 = c1 @ normals.T
                p2 = c2 @ normals.T
                low = np.maximum(p1 - a, p2 - a)
                high = np.minimum(p1 + a, p2 + a)
                gaps = high - low
                max_gap = np.max(gaps)
                if max_gap < 0:
                    overlap_pen += -max_gap
        
        total_pen = viol_contain + overlap_pen
        return S + 100.0 * total_pen

    best_S = float('inf')
    best_pos = None
    
    # Multiple restarts
    num_restarts = 20
    for r in range(num_restarts):
        if time.time() - start_time > timeout - 60:
            break
            
        # Initial guess
        init_S = 5.0 + r * 0.1
        init_pos = []
        # Center
        init_pos.append([0.0, 0.0])
        # Ring 1
        for i in range(6):
            angle = i * np.pi / 3
            r_dist = np.sqrt(3)
            init_pos.append([r_dist * np.cos(angle), r_dist * np.sin(angle)])
        # Ring 2 (randomized)
        for i in range(4):
            angle = np.random.uniform(0, 2*np.pi)
            r_dist = np.random.uniform(2.5, 4.0)
            init_pos.append([r_dist * np.cos(angle), r_dist * np.sin(angle)])
            
        init_x = np.array([init_S] + [p[0] for p in init_pos] + [p[1] for p in init_pos])
        
        # Optimize
        try:
            res = minimize(objective, init_x, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        except Exception:
            continue
            
        S = res.x[0]
        pos = np.array(res.x[1:]).reshape(-1, 2)
        
        if is_valid(S, pos):
            if S < best_S:
                best_S = S
                best_pos = pos.copy()
                
        # If we found a very good solution, we can stop early?
        # Theoretical lower bound is around 4.something.
        # If best_S < 4.5, it's likely near optimal.
        if best_S < 4.2:
            break

    if best_pos is None:
        # Fallback to a known valid, though not optimal, configuration
        # Simple grid
        S = 8.0
        pos = np.array([
            [0, 0],
            [-2.5, 0], [2.5, 0],
            [-1.25, 2.17], [1.25, 2.17],
            [-1.25, -2.17], [1.25, -2.17],
            [-3.75, 2.17], [3.75, 2.17],
            [-3.75, -2.17], [3.75, -2.17]
        ])
    else:
        S = best_S
        pos = best_pos

    inner = np.column_stack((pos[:, 0], pos[:, 1], np.zeros(n_hex)))
    outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, float(S)
