"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 110.0  # Keep 10s margin for safety

    sqrt3 = np.sqrt(3)

    # Initial configuration: 12 hexagons in a symmetric cluster
    # Flat-top hexagons
    # Layout:
    #       1
    #     1 1 1
    #   1 1 1 1
    #     1 1 1
    #       1
    # Coordinates for this layout:
    # Row 4 (top):    y = sqrt3, x = 0
    # Row 3:          y = sqrt3/2, x = -1, 0, 1
    # Row 2 (mid):    y = 0, x = -1.5, -0.5, 0.5, 1.5
    # Row 1:          y = -sqrt3/2, x = -1, 0, 1
    # Row 0 (bot):    y = -sqrt3, x = 0
    
    init_pos = [
        [0.0, sqrt3],
        [-1.0, sqrt3/2], [0.0, sqrt3/2], [1.0, sqrt3/2],
        [-1.5, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0],
        [-1.0, -sqrt3/2], [0.0, -sqrt3/2], [1.0, -sqrt3/2],
        [0.0, -sqrt3]
    ]
    
    x0 = np.array([p[0] for p in init_pos])
    y0 = np.array([p[1] for p in init_pos])
    v0 = np.concatenate([x0, y0])
    
    # Relative vertices for unit flat-top hexagon
    rel_x = np.array([1, 0.5, -0.5, -1, -0.5, 0.5])
    rel_y = np.array([0, sqrt3/2, sqrt3/2, 0, -sqrt3/2, -sqrt3/2])

    def get_vertices(x, y):
        # x, y: (N,)
        # Returns vx, vy: (N, 6)
        vx = x[:, np.newaxis] + rel_x[np.newaxis, :]
        vy = y[:, np.newaxis] + rel_y[np.newaxis, :]
        return vx, vy

    def objective(params):
        if time.time() - start_time > timeout:
            return 1e9
            
        x = params[:N]
        y = params[N:]
        
        vx, vy = get_vertices(x, y)
        
        # Flatten
        vx_flat = vx.flatten()
        vy_flat = vy.flatten()
        
        # Constraints for A (apothem of outer hexagon)
        # 1. |x| <= A
        c1 = np.abs(vx_flat)
        
        # 2. |0.5*x + sqrt(3)/2*y| <= A
        c2 = np.abs(0.5 * vx_flat + (sqrt3/2) * vy_flat)
        
        # 3. |-0.5*x + sqrt(3)/2*y| <= A
        c3 = np.abs(-0.5 * vx_flat + (sqrt3/2) * vy_flat)
        
        A = np.maximum.reduce([c1, c2, c3]).max()
        
        return A

    best_params = v0
    best_score = objective(v0)
    
    try:
        from scipy.optimize import minimize
        
        # Powell method is good for this
        res = minimize(objective, v0, method='Powell', options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-7})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
            
        # Try Nelder-Mead if time permits
        if time.time() - start_time < timeout - 30:
            res_nm = minimize(objective, v0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-7})
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x
                
    except Exception:
        pass

    x_opt = best_params[:N]
    y_opt = best_params[N:]
    
    # Calculate final A and S
    vx, vy = get_vertices(x_opt, y_opt)
    vx_flat = vx.flatten()
    vy_flat = vy.flatten()
    c1 = np.abs(vx_flat)
    c2 = np.abs(0.5 * vx_flat + (sqrt3/2) * vy_flat)
    c3 = np.abs(-0.5 * vx_flat + (sqrt3/2) * vy_flat)
    A_opt = np.maximum.reduce([c1, c2, c3]).max()
    
    S_opt = A_opt * 2.0 / sqrt3
    
    inner = []
    for i in range(N):
        inner.append([x_opt[i], y_opt[i], 0.0])
        
    return inner, (0.0, 0.0), S_opt, 0.0
# EVOLVE-BLOCK-END
