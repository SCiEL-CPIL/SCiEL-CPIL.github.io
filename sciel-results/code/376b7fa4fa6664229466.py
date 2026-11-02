import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 140.0

def objective(params):
    # params[0:36] = 12*(x, y, angle_deg) for inner hexagons
    # params[36] = outer hexagon rotation (radians)
    n = N
    outer_rot = params[36]
    
    all_verts = np.zeros((n * 6, 2))
    idx = 0
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.radians(params[3*i+2])
        for j in range(6):
            a = ang + j * np.pi / 3
            all_verts[idx, 0] = cx + np.cos(a)
            all_verts[idx, 1] = cy + np.sin(a)
            idx += 1
    
    max_proj = 0.0
    for k in range(6):
        a = outer_rot + k * np.pi / 3
        nx = np.cos(a)
        ny = np.sin(a)
        proj = np.max(all_verts[:, 0] * nx + all_verts[:, 1] * ny)
        max_proj = max(max_proj, proj)
    
    return max_proj / (np.sqrt(3) / 2)

def construct():
    t0 = time.time()
    np.random.seed(42)
    
    # Initial guess using hex lattice
    axial_coords = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    
    x0 = np.zeros(37)
    for i, (q, r) in enumerate(axial_coords):
        x0[3*i] = np.sqrt(3) * (q + r / 2)
        x0[3*i+1] = 1.5 * r
        x0[3*i+2] = 0.0
    x0[36] = 0.0
    
    best_params = x0.copy()
    best_val = objective(x0)
    
    # Nelder-Mead optimization
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
    if result.fun < best_val:
        best_val = result.fun
        best_params = result.x
    
    # Random restarts
    for restart in range(30):
        if time.time() > DEADLINE - 15:
            break
        # Different initial spreads
        scale = 1.0 + restart * 0.3
        x_r = x0 + np.random.randn(37) * 0.3
        x_r[36] = np.random.uniform(-np.pi/6, np.pi/6)
        
        res = minimize(objective, x_r, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], np.degrees(best_params[3*i+2])])
    
    outer_rot_deg = np.degrees(best_params[36])
    return inner, (0.0, 0.0), best_val, outer_rot_deg
