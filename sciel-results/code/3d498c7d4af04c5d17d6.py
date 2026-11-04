"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    np.random.seed(42)
    
    # Simulated Annealing Approach
    # Variables: w, x_1..x_N, y_1..y_N, r_1..r_N
    # Total dims: 1 + 3*N
    
    def evaluate(v):
        w = v[0]
        if w <= 0.1 or w >= 1.9:
            return -np.inf
        h = 2.0 - w
        
        xs = v[1:1+N]
        ys = v[1+N:1+2*N]
        rs = v[1+2*N:1+3*N]
        
        # Check boundary
        penalty = 0.0
        for i in range(N):
            if rs[i] < 0:
                return -np.inf
            if xs[i] < rs[i] or xs[i] > w - rs[i]:
                penalty += (max(0, rs[i] - xs[i]) + max(0, xs[i] - (w - rs[i]))) ** 2
            if ys[i] < rs[i] or ys[i] > h - rs[i]:
                penalty += (max(0, rs[i] - ys[i]) + max(0, ys[i] - (h - rs[i]))) ** 2
        
        # Check overlaps
        for i in range(N):
            for j in range(i+1, N):
                dist = np.sqrt((xs[i] - xs[j])**2 + (ys[i] - ys[j])**2)
                min_dist = rs[i] + rs[j]
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        if penalty > 1e-4:
            return -np.inf - penalty
            
        return np.sum(rs)

    def step(current, temp):
        # Random perturbation
        scale = 0.1 * temp
        new_v = current.copy()
        
        # Perturb w
        new_v[0] += np.random.normal(0, scale * 0.2)
        new_v[0] = np.clip(new_v[0], 0.5, 1.5)
        
        # Perturb positions and radii
        for i in range(N):
            idx_x = 1 + i
            idx_y = 1 + N + i
            idx_r = 1 + 2*N + i
            
            new_v[idx_x] += np.random.normal(0, scale)
            new_v[idx_y] += np.random.normal(0, scale)
            new_v[idx_r] += np.random.normal(0, scale * 0.5)
            
            # Clip radii to be positive
            new_v[idx_r] = max(0.01, new_v[idx_r])
            
        return new_v

    best_score = -np.inf
    best_v = None
    
    # Initialize with a few different starting points
    start_configs = []
    
    # Config 1: Grid (Square-ish)
    w_init = 1.0
    h_init = 1.0
    cols = 5
    rows = 5
    v_init = np.zeros(1 + 3*N)
    v_init[0] = w_init
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = (c + 0.5) * w_init / cols
            y = (r + 0.5) * h_init / rows
            r_rad = 0.4 / cols
            v_init[1 + count] = x
            v_init[1 + N + count] = y
            v_init[1 + 2*N + count] = r_rad
            count += 1
    start_configs.append(v_init)
    
    # Config 2: Rectangular (Wide)
    w_init = 1.4
    h_init = 0.6
    cols = 7
    rows = 3
    v_init = np.zeros(1 + 3*N)
    v_init[0] = w_init
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = (c + 0.5) * w_init / cols
            y = (r + 0.5) * h_init / rows
            r_rad = 0.4 / cols
            v_init[1 + count] = x
            v_init[1 + N + count] = y
            v_init[1 + 2*N + count] = r_rad
            count += 1
    start_configs.append(v_init)

    # Config 3: Rectangular (Tall)
    w_init = 0.6
    h_init = 1.4
    cols = 3
    rows = 7
    v_init = np.zeros(1 + 3*N)
    v_init[0] = w_init
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = (c + 0.5) * w_init / cols
            y = (r + 0.5) * h_init / rows
            r_rad = 0.4 / cols
            v_init[1 + count] = x
            v_init[1 + N + count] = y
            v_init[1 + 2*N + count] = r_rad
            count += 1
    start_configs.append(v_init)

    start_time = time.time()
    
    for v0 in start_configs:
        current_v = v0.copy()
        current_score = evaluate(current_v)
        if current_score == -np.inf:
             for i in range(N):
                 current_v[1 + 2*N + i] *= 0.5
             current_score = evaluate(current_v)
        
        best_v_local = current_v.copy()
        best_score_local = current_score
        
        T = 1.0
        T_min = 1e-5
        alpha = 0.995
        max_iter = 10000
        
        for it in range(max_iter):
            if time.time() - start_time > 100:
                break
                
            new_v = step(current_v, T)
            new_score = evaluate(new_v)
            
            if new_score > current_score:
                current_v = new_v
                current_score = new_score
                if current_score > best_score_local:
                    best_score_local = current_score
                    best_v_local = current_v.copy()
            else:
                delta = new_score - current_score
                if delta < 0:
                    prob = np.exp(delta / T)
                    if np.random.rand() < prob:
                        current_v = new_v
                        current_score = new_score
                else:
                    current_v = new_v
                    current_score = new_score
            
            T *= alpha
            if T < T_min:
                break
                
        if best_score_local > best_score:
            best_score = best_score_local
            best_v = best_v_local.copy()
            
    # Final polish with L-BFGS if we found a good solution
    if best_v is not None:
        def objective_polish(v):
            return -evaluate(v)
        
        res = minimize(objective_polish, best_v, method='L-BFGS-B', options={'maxiter': 100})
        if res.fun < -best_score:
            # Check validity of polished result
            if evaluate(res.x) != -np.inf:
                best_v = res.x
                best_score = -res.fun

    # Extract result
    if best_v is None:
        # Fallback
        return np.array([[(i+0.5)/5, (j+0.5)/5, 0.1] for i in range(5) for j in range(5)][:N])
        
    w = best_v[0]
    h = 2.0 - w
    xs = best_v[1:1+N]
    ys = best_v[1+N:1+2*N]
    rs = best_v[1+2*N:1+3*N]
    
    # Ensure strict validity
    valid = True
    for i in range(N):
        if xs[i] < rs[i] - 1e-7 or xs[i] > w - rs[i] + 1e-7:
            valid = False
        if ys[i] < rs[i] - 1e-7 or ys[i] > h - rs[i] + 1e-7:
            valid = False
    for i in range(N):
        for j in range(i+1, N):
            dist = np.sqrt((xs[i]-xs[j])**2 + (ys[i]-ys[j])**2)
            if dist < rs[i] + rs[j] - 1e-7:
                valid = False
    
    if not valid:
        # Shrink radii slightly to ensure validity
        rs = rs * 0.99

    return np.column_stack([xs, ys, rs])
# EVOLVE-BLOCK-END
