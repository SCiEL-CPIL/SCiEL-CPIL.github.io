import numpy as np
from scipy.optimize import minimize
import time

N = 21

def construct():
    # Helper to generate a grid configuration
    def make_grid(n_cols, n_rows, W, H):
        # n_cols * n_rows >= N
        x = []
        y = []
        for i in range(n_cols):
            for j in range(n_rows):
                x.append((i + 0.5) * W / n_cols)
                y.append((j + 0.5) * H / n_rows)
        return np.array(x[:N]), np.array(y[:N])

    def evaluate_config(x, y, r, W, H):
        # Returns sum of radii if valid, else -inf
        if W <= 0 or H <= 0:
            return -np.inf
        # Perimeter constraint: 2(W+H) = 4 => W+H = 2
        if abs(2*(W+H) - 4) > 1e-6:
            return -np.inf
        for i in range(N):
            if r[i] <= 0:
                return -np.inf
            if x[i] - r[i] < -1e-6 or x[i] + r[i] > W + 1e-6:
                return -np.inf
            if y[i] - r[i] < -1e-6 or y[i] + r[i] > H + 1e-6:
                return -np.inf
            for j in range(i+1, N):
                dist_sq = (x[i]-x[j])**2 + (y[i]-y[j])**2
                rad_sum = r[i] + r[j]
                if dist_sq < rad_sum**2 - 1e-6:
                    return -np.inf
        return np.sum(r)

    def optimize(x0, y0, r0, W0, H0):
        # SLSQP optimization
        def obj(v):
            x = v[:N]
            y = v[N:2*N]
            r = v[2*N:3*N]
            W = v[3*N]
            H = v[3*N+1]
            return -np.sum(r)
            
        def jac(v):
            g = np.zeros_like(v)
            g[2*N:3*N] = -1.0
            return g
            
        constraints = []
        # Perimeter constraint: 2(W+H) - 4 = 0
        constraints.append({'type': 'eq', 'fun': lambda v: 2*(v[3*N] + v[3*N+1]) - 4})
        
        # Circle-Circle non-overlap
        for i in range(N):
            for j in range(i+1, N):
                constraints.append({
                    'type': 'ineq', 
                    'fun': lambda v, i=i, j=j: (v[i]-v[N+j])**2 + (v[N+i]-v[2*N+j])**2 - (v[2*N+i]+v[2*N+j])**2
                })
                
        # Circle-Wall constraints
        for i in range(N):
            # Left
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[i] - v[2*N+i]})
            # Right
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N] - v[i] - v[2*N+i]})
            # Bottom
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[N+i] - v[2*N+i]})
            # Top
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[3*N+1] - v[N+i] - v[2*N+i]})
            # Positive radius
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*N+i]})
            
        # Positive dimensions
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N]})
        constraints.append({'type': 'ineq', 'fun': lambda v: v[3*N+1]})

        x_start = np.concatenate([x0, y0, r0, [W0, H0]])
        
        try:
            res = minimize(obj, x_start, jac=jac, method='SLSQP', constraints=constraints, 
                           options={'maxiter': 100, 'ftol': 1e-6})
            if res.success:
                x_final = res.x[:N]
                y_final = res.x[N:2*N]
                r_final = res.x[2*N:3*N]
                W_final = res.x[3*N]
                H_final = res.x[3*N+1]
                score = evaluate_config(x_final, y_final, r_final, W_final, H_final)
                if score > -np.inf:
                    return x_final, y_final, r_final, W_final, H_final, score
        except Exception:
            pass
        return None

    # Generate multiple initial candidates
    candidates = []
    
    # 1. Square grid (1x1)
    x_sq, y_sq = make_grid(5, 5, 1.0, 1.0) # 5x5=25 >= 21
    r_sq = np.full(N, 0.05)
    res = optimize(x_sq, y_sq, r_sq, 1.0, 1.0)
    if res: candidates.append(res)

    # 2. Wide rectangle (2x1)
    x_wide, y_wide = make_grid(7, 3, 2.0, 1.0) # 7x3=21 exactly
    r_wide = np.full(N, 0.05)
    res = optimize(x_wide, y_wide, r_wide, 2.0, 1.0)
    if res: candidates.append(res)
    
    # 3. Tall rectangle (1x2)
    x_tall, y_tall = make_grid(3, 7, 1.0, 2.0) # 3x7=21 exactly
    r_tall = np.full(N, 0.05)
    res = optimize(x_tall, y_tall, r_tall, 1.0, 2.0)
    if res: candidates.append(res)
    
    # 4. Jittered square
    x_jit = x_sq + np.random.normal(0, 0.05, N)
    y_jit = y_sq + np.random.normal(0, 0.05, N)
    r_jit = np.full(N, 0.05)
    res = optimize(x_jit, y_jit, r_jit, 1.0, 1.0)
    if res: candidates.append(res)
    
    # 5. Jittered wide
    x_jit2 = x_wide + np.random.normal(0, 0.05, N)
    y_jit2 = y_wide + np.random.normal(0, 0.05, N)
    r_jit2 = np.full(N, 0.05)
    res = optimize(x_jit2, y_jit2, r_jit2, 2.0, 1.0)
    if res: candidates.append(res)

    if not candidates:
        # Fallback: Simple valid configuration
        # 5x5 grid in 1x1 box, radius 0.09
        x, y = make_grid(5, 5, 1.0, 1.0)
        r = np.full(N, 0.09)
        # Check validity of fallback
        # Dist between neighbors is 0.2, sum of radii is 0.18 < 0.2. OK.
        # Wall dist is 0.1, radius 0.09 < 0.1. OK.
        return np.column_stack([x, y, r])

    # Select best candidate
    best_candidate = max(candidates, key=lambda c: c[5])
    best_x, best_y, best_r, best_W, best_H, best_score = best_candidate
    
    # Try to improve the best candidate with another round of optimization from a slightly perturbed state
    x_pert = best_x + np.random.normal(0, 0.01, N)
    y_pert = best_y + np.random.normal(0, 0.01, N)
    r_pert = best_r.copy()
    res = optimize(x_pert, y_pert, r_pert, best_W, best_H)
    if res:
        if res[5] > best_score:
            best_x, best_y, best_r, best_W, best_H, best_score = res

    result = np.zeros((N, 3))
    result[:, 0] = best_x
    result[:, 1] = best_y
    result[:, 2] = best_r
    
    return result
