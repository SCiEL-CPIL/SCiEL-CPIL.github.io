import numpy as np
from scipy.optimize import linprog, minimize
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Base vertices of unit hexagon (angle 0: vertex at 1,0)
    angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    base_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    def solve_lp(theta, alpha):
        # Compute vertex offsets for inner hexagons
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        offsets = base_vertices @ R.T  # (6, 2)
        
        # Compute normals for outer hexagon
        normal_angles = theta + np.pi/6 + 2*np.pi*np.arange(6)/6
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        
        n_vars = 1 + 2*N
        c = np.zeros(n_vars)
        c[0] = 1.0
        
        A_ub = []
        b_ub = []
        factor = np.sqrt(3) / 2.0
        
        for k in range(N):
            x_idx = 1 + 2*k
            y_idx = 2 + 2*k
            
            for j in range(6):
                vx, vy = offsets[j]
                for i in range(6):
                    nx, ny = normals[i]
                    
                    row1 = np.zeros(n_vars)
                    row1[0] = -factor
                    row1[x_idx] = nx
                    row1[y_idx] = ny
                    A_ub.append(row1)
                    b_ub.append(-(nx*vx + ny*vy))
                    
                    row2 = np.zeros(n_vars)
                    row2[0] = -factor
                    row2[x_idx] = -nx
                    row2[y_idx] = -ny
                    A_ub.append(row2)
                    b_ub.append(nx*vx + ny*vy)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] + [(None, None)] * (2*N)
        
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        return None

    best_S = float('inf')
    best_params = None
    
    # Grid search
    theta_vals = np.linspace(0, np.pi/3, 10)
    alpha_vals = np.linspace(0, np.pi/3, 10)
    
    for theta in theta_vals:
        if time.time() > deadline:
            break
        for alpha in alpha_vals:
            if time.time() > deadline:
                break
            res = solve_lp(theta, alpha)
            if res is not None:
                S_val = res[0]
                if S_val < best_S:
                    best_S = S_val
                    best_params = res
                    best_theta = theta
                    best_alpha = alpha

    # Refinement using Nelder-Mead on theta, alpha
    if time.time() < deadline - 10:
        def cost_func(params):
            t, a = params
            res = solve_lp(t, a)
            if res is None:
                return 1e6
            return res[0]
        
        # Start from best grid point
        init_t_a = [best_theta, best_alpha]
        res_opt = minimize(cost_func, init_t_a, method='Nelder-Mead', options={'maxiter': 100})
        
        if res_opt.fun < best_S:
            best_S = res_opt.fun
            # Re-solve LP to get positions
            res = solve_lp(res_opt.x[0], res_opt.x[1])
            if res is not None:
                best_params = res
                best_theta = res_opt.x[0]
                best_alpha = res_opt.x[1]

    if best_params is None:
        inner = [[0.0, 0.0, 0.0] for _ in range(N)]
        return inner, (0.0, 0.0), 10.0, 0.0

    positions = best_params[1:].reshape(N, 2)
    inner = [[float(p[0]), float(p[1]), float(np.degrees(best_alpha))] for p in positions]
    
    return inner, (0.0, 0.0), float(best_S), float(np.degrees(best_theta))
