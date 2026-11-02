import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_dist_to_outside(px, py, cx, cy, side, angle_deg):
    """Signed distance from point to hexagon boundary (negative if inside)."""
    a = np.radians(angle_deg)
    dx, dy = px - cx, py - cy
    ca, sa = np.cos(-a), np.sin(-a)
    lx = ca * dx + sa * dy
    ly = -sa * dx + ca * dy
    r = side * np.sqrt(3) / 2
    # 6 edge normals for flat-top hex at angle 0
    angles = np.pi/6 + np.arange(6) * np.pi/3
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)
    dots = lx * cos_a + ly * sin_a
    return np.max(dots) - r

def _hex_vertices(cx, cy, side, angle_deg):
    a = np.radians(angle_deg)
    angles = a + np.pi/6 + np.arange(6) * np.pi/3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _objective(params):
    inner = [(params[3*i], params[3*i+1], params[3*i+2]) for i in range(N)]
    ocx, ocy = params[36], params[37]
    oside = max(params[38], 2.0)
    oangle = params[39]
    
    pen = 0.0
    # Containment
    for i in range(N):
        cx, cy, a = inner[i]
        verts = _hex_vertices(cx, cy, 1.0, a)
        for v in verts:
            d = _hex_dist_to_outside(v[0], v[1], ocx, ocy, oside, oangle)
            if d > 0:
                pen += d * d
    
    # Overlap: check vertex containment between pairs
    for i in range(N):
        cx1, cy1, a1 = inner[i]
        v1 = _hex_vertices(cx1, cy1, 1.0, a1)
        for j in range(i+1, N):
            cx2, cy2, a2 = inner[j]
            v2 = _hex_vertices(cx2, cy2, 1.0, a2)
            # Check vertices of 1 in 2
            for v in v1:
                d = _hex_dist_to_outside(v[0], v[1], cx2, cy2, 1.0, a2)
                if d < 0:
                    pen += d * d
            # Check vertices of 2 in 1
            for v in v2:
                d = _hex_dist_to_outside(v[0], v[1], cx1, cy1, 1.0, a1)
                if d < 0:
                    pen += d * d
    
    return oside + 50.0 * pen

def construct():
    t0 = time.time()
    deadline = t0 + 140.0
    
    # 3x4 grid, flat-top
    cols, rows = 4, 3
    dx = 2.0
    dy = np.sqrt(3)
    
    params = np.zeros(40)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            params[3*idx] = (c - (cols-1)/2.0) * dx
            params[3*idx+1] = (r - (rows-1)/2.0) * dy
            params[3*idx+2] = 0.0
            idx += 1
    params[36:38] = [0.0, 0.0]
    params[38] = 4.5
    params[39] = 0.0
    
    best_params = params.copy()
    best_val = _objective(best_params)
    
    # Simulated annealing
    T = 0.5
    T_min = 1e-5
    cooling = 0.997
    rng = np.random.default_rng(123)
    
    n_iter = 0
    while T > T_min and time.time() < deadline - 10:
        n_iter += 1
        new_params = best_params.copy()
        
        # Perturb 1-2 random inner hexes
        n_pert = rng.integers(1, 3)
        for _ in range(n_pert):
            i = rng.integers(0, N)
            step = 0.2 * np.sqrt(T)
            new_params[3*i] += rng.normal(0, step)
            new_params[3*i+1] += rng.normal(0, step)
            new_params[3*i+2] += rng.normal(0, step * 20)
        
        if rng.random() < 0.05:
            new_params[36] += rng.normal(0, 0.1)
            new_params[37] += rng.normal(0, 0.1)
            new_params[38] += rng.normal(0, 0.1)
            new_params[39] += rng.normal(0, 0.5)
        
        new_val = _objective(new_params)
        
        if new_val < best_val or rng.random() < np.exp(-(new_val - best_val) / max(T, 1e-10)):
            best_params = new_params
            best_val = new_val
        
        T *= cooling
    
    # Local refinement
    if time.time() < deadline - 5:
        try:
            res = minimize(_objective, best_params, method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < best_val:
                best_params = res.x
                best_val = res.fun
        except:
            pass
    
    inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
    return inner, (best_params[36], best_params[37]), max(best_params[38], 2.0), best_params[39]
# EVOLVE-BLOCK-END
