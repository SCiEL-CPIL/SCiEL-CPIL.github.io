"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Precompute unit hexagon vertices (flat-top, center at 0,0, side 1)
    angles = np.deg2rad(np.arange(0, 360, 60))
    unit_hex_vertices = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Outer hexagon normals (for flat-top, inradius r_out)
    outer_normals = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Initial guess: Hexagonal cluster
    def get_initial_params():
        pos = [(0.0, 0.0)]
        d1 = np.sqrt(3.0)
        for k in range(6):
            angle = k * np.pi / 3
            pos.append((d1 * np.cos(angle), d1 * np.sin(angle)))
        d2 = 2 * np.sqrt(3.0)
        for k in range(5):
            angle = k * np.pi / 3
            pos.append((d2 * np.cos(angle), d2 * np.sin(angle)))
            
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        thetas = [0.0] * N
        
        # Estimate S
        # Max x is approx 3.46 + 1 = 4.46
        # r_out = S * sqrt(3)/2
        # 4.46 <= r_out
        # S >= 4.46 * 2 / sqrt(3) = 5.15
        S_init = 5.5
        
        params = np.zeros(N * 3 + 1)
        params[0::3] = xs
        params[1::3] = ys
        params[2::3] = thetas
        params[-1] = S_init
        return params

    p0 = get_initial_params()
    
    def unpack(params):
        S = params[-1]
        coords = params[:-1].reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        thetas = coords[:, 2]
        return xs, ys, thetas, S
        
    def objective(params):
        S = params[-1]
        coords = params[:-1].reshape(N, 3)
        xs = coords[:, 0]
        ys = coords[:, 1]
        thetas = coords[:, 2]
        
        # 1. Overlap Penalty
        dx = xs[:, None] - xs[None, :]
        dy = ys[:, None] - ys[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        d_min = np.sqrt(3.0)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d_subset = d[mask]
        
        violations = np.maximum(0, d_min - d_subset)
        overlap_pen = np.sum(violations**2)
        
        # 2. Boundary Penalty
        r_out = S * np.sqrt(3.0) / 2.0
        
        c = np.cos(thetas)
        s = np.sin(thetas)
        
        vx0 = unit_hex_vertices[:, 0]
        vy0 = unit_hex_vertices[:, 1]
        
        vx = c[:, None] * vx0[None, :] - s[:, None] * vy0[None, :] + xs[:, None]
        vy = s[:, None] * vx0[None, :] + c[:, None] * vy0[None, :] + ys[:, None]
        
        verts = np.stack([vx, vy], axis=-1)
        verts_flat = verts.reshape(-1, 2)
        
        proj = verts_flat @ outer_normals.T
        violations_bound = np.maximum(0, proj - r_out)
        bound_pen = np.sum(violations_bound**2)
        
        w_overlap = 1000.0
        w_bound = 1000.0
        
        return S + w_overlap * overlap_pen + w_bound * bound_pen

    best_params = p0
    best_val = objective(p0)
    
    n_restarts = 20
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            p = p0
        else:
            p = p0 + np.random.normal(0, 0.2, p0.shape)
            p[-1] = p0[-1] + np.random.uniform(-0.5, 0.5)
            
        try:
            res = minimize(objective, p, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Final calculation of exact S required for best configuration
    xs, ys, thetas, _ = unpack(best_params)
    
    c = np.cos(thetas)
    s = np.sin(thetas)
    
    vx0 = unit_hex_vertices[:, 0]
    vy0 = unit_hex_vertices[:, 1]
    
    vx = c[:, None] * vx0[None, :] - s[:, None] * vy0[None, :] + xs[:, None]
    vy = s[:, None] * vx0[None, :] + c[:, None] * vy0[None, :] + ys[:, None]
    
    verts = np.stack([vx, vy], axis=-1)
    verts_flat = verts.reshape(-1, 2)
    
    proj = verts_flat @ outer_normals.T
    max_proj = np.max(proj)
    
    # r_out = max_proj
    # S = r_out * 2 / sqrt(3)
    S_final = max_proj * 2.0 / np.sqrt(3.0)
    
    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(np.rad2deg(thetas[i]))])
        
    return inner, (0.0, 0.0), float(S_final), 0.0
# EVOLVE-BLOCK-END
