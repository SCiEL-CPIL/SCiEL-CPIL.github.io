import numpy as np
import time
from scipy.optimize import linprog

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin for 120s limit
    
    n = 21
    
    # Initial configuration: 5x5 grid in 1x1 square, remove 4 corners
    # 5x5 has 25. We need 21.
    # Removing corners (0,0), (0,4), (4,0), (4,4)
    
    W_init = 1.0
    H_init = 1.0
    
    centers = []
    for i in range(5):
        for j in range(5):
            if (i == 0 and j == 0) or (i == 0 and j == 4) or (i == 4 and j == 0) or (i == 4 and j == 4):
                continue
            x = (i + 0.5) * 0.2
            y = (j + 0.5) * 0.2
            centers.append([x, y])
            
    centers = np.array(centers)
    assert len(centers) == 21
    
    def solve_lp(centers, W, H):
        # Variables: r_0..r_20, W, H
        # Minimize -sum(r)
        c = np.zeros(n + 2)
        c[:n] = -1.0
        
        # Constraints:
        # 1. W + H = 2
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([2.0])
        
        A_ub = []
        b_ub = []
        
        # Pairwise constraints: r_i + r_j <= dist_ij
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(centers[i] - centers[j])
                row = np.zeros(n + 2)
                row[i] = 1.0
                row[j] = 1.0
                A_ub.append(row)
                b_ub.append(d)
                
        # Wall constraints
        for i in range(n):
            x, y = centers[i]
            
            # r_i <= x
            row = np.zeros(n + 2)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(x)
            
            # r_i <= W - x  => r_i - W <= -x
            row = np.zeros(n + 2)
            row[i] = 1.0
            row[n] = -1.0  # W
            A_ub.append(row)
            b_ub.append(-x)
            
            # r_i <= y
            row = np.zeros(n + 2)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(y)
            
            # r_i <= H - y  => r_i - H <= -y
            row = np.zeros(n + 2)
            row[i] = 1.0
            row[n+1] = -1.0  # H
            A_ub.append(row)
            b_ub.append(-y)
            
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds: r_i >= 0, W >= 0, H >= 0
        bounds = [(0, None)] * (n + 2)
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            
            if res.status == 0:
                radii = res.x[:n]
                W_opt = res.x[n]
                H_opt = res.x[n+1]
                return radii, W_opt, H_opt, res.fun
            else:
                return None, W, H, -np.inf
        except Exception:
            return None, W, H, -np.inf

    # Initial solve
    radii, W, H, score = solve_lp(centers, W_init, H_init)
    
    # Fallback if LP fails
    if radii is None:
        r_init = 0.1
        return np.column_stack([centers, np.full(n, r_init * 0.999)])

    best_score = -score
    best_radii = radii
    best_W = W
    best_H = H
    best_centers = centers.copy()
    
    # Local search
    np.random.seed(42)
    step_size = 0.02
    
    while time.time() - start_time < timeout:
        # Pick a random circle to move
        idx = np.random.randint(n)
        
        # Small perturbation
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_centers = best_centers.copy()
        new_centers[idx][0] += dx
        new_centers[idx][1] += dy
        
        new_radii, new_W, new_H, new_score = solve_lp(new_centers, best_W, best_H)
        
        if new_radii is not None:
            if -new_score > best_score:
                best_score = -new_score
                best_radii = new_radii
                best_W = new_W
                best_H = new_H
                best_centers = new_centers
                # Optionally reduce step size on success? Or keep it?
                # step_size = max(0.001, step_size * 0.95)
            else:
                # No improvement, maybe increase step size slightly to explore more?
                # step_size = min(0.05, step_size * 1.05)
                pass
        else:
            # Infeasible, reduce step size
            step_size = max(0.001, step_size * 0.5)

    # Scale down radii slightly to ensure strict disjointness for evaluator
    scale = 0.9999
    final_radii = best_radii * scale
    
    result = np.column_stack([best_centers, final_radii])
    
    return result
