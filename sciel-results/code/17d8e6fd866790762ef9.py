import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Vertices of a unit hexagon (side 1) centered at origin, angle 0
hex_verts = np.array([
    [1.0, 0.0],
    [0.5, np.sqrt(3)/2],
    [-0.5, np.sqrt(3)/2],
    [-1.0, 0.0],
    [-0.5, -np.sqrt(3)/2],
    [0.5, -np.sqrt(3)/2]
])

# Normals for the outer hexagon (rot 0)
angles = np.deg2rad([30, 90, 150, 210, 270, 330])
normals = np.column_stack([np.cos(angles), np.sin(angles)])
inv_apothem_factor = 2.0 / np.sqrt(3)

def get_vertices(xs, ys):
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = hex_verts + np.array([xs[i], ys[i]])
    return verts

def calc_L(xs, ys):
    verts = get_vertices(xs, ys)
    projs = np.einsum('ijk,jk->ij', verts, normals)
    max_proj = np.max(projs)
    return max_proj * inv_apothem_factor

def calc_overlap_penalty(xs, ys, threshold=np.sqrt(3)):
    penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < threshold:
                penalty += (threshold - dist)**2
    return penalty

def objective(vars):
    xs = vars[:N]
    ys = vars[N:]
    L = calc_L(xs, ys)
    pen = calc_overlap_penalty(xs, ys)
    return L + 100.0 * pen

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    best_L = float('inf')
    best_config = None
    
    rng = np.random.default_rng(42)
    
    # Initial guess: Hex cluster of 7 + 4
    # 7 cluster: center (0,0), 6 around
    xs_init = [0.0]
    ys_init = [0.0]
    for k in range(6):
        angle = k * np.pi / 3
        xs_init.append(np.sqrt(3) * np.cos(angle))
        ys_init.append(np.sqrt(3) * np.sin(angle))
    # Add 4 more in a rough line
    for k in range(4):
        xs_init.append(2*np.sqrt(3) + k*np.sqrt(3))
        ys_init.append(0.0)
        
    initial_vars = np.concatenate([xs_init, ys_init])
    
    # Try the structured init first
    try:
        res = minimize(objective, initial_vars, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            xs_opt = res.x[:N]
            ys_opt = res.x[N:]
            L_opt = calc_L(xs_opt, ys_opt)
            pen_opt = calc_overlap_penalty(xs_opt, ys_opt)
            if pen_opt < 1e-6:
                best_L = L_opt
                best_config = (xs_opt, ys_opt)
    except Exception:
        pass

    # Random restarts
    for restart in range(40):
        if time.time() > deadline - 5.0:
            break
            
        xs = rng.uniform(-4, 4, N)
        ys = rng.uniform(-4, 4, N)
        vars0 = np.concatenate([xs, ys])
        
        try:
            res = minimize(objective, vars0, method='Nelder-Mead', 
                           options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-7})
            
            if res.success:
                xs_opt = res.x[:N]
                ys_opt = res.x[N:]
                L_opt = calc_L(xs_opt, ys_opt)
                pen_opt = calc_overlap_penalty(xs_opt, ys_opt)
                
                if pen_opt < 1e-6: # Valid enough
                    if L_opt < best_L:
                        best_L = L_opt
                        best_config = (xs_opt, ys_opt)
        except Exception:
            pass

    if best_config is None:
        # Fallback
        return [[0.0, 0.0, 0.0] for _ in range(N)], (0.0, 0.0), 10.0, 0.0

    xs_opt, ys_opt = best_config
    inner = [[float(x), float(y), 0.0] for x, y in zip(xs_opt, ys_opt)]
    return inner, (0.0, 0.0), float(best_L), 0.0
