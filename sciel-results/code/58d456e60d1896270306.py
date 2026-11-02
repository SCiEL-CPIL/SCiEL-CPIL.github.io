import math
import numpy as np
import time
import random

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    random.seed(12345)
    np.random.seed(12345)
    
    # Initial configuration: compact hexagonal arrangement
    init_axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    
    positions = np.zeros((N, 2))
    for i, (q, r) in enumerate(init_axial):
        positions[i, 0] = math.sqrt(3) * (q + r / 2.0)
        positions[i, 1] = 1.5 * r
    angles = np.zeros(N)
    
    def compute_S(pos, ang):
        """Compute outer hexagon side length. Returns (S, centroid, D, beta)."""
        # Compute all 66 vertices
        verts = np.zeros((N * 6, 2))
        for i in range(N):
            a = math.radians(ang[i])
            for k in range(6):
                v_ang = a + k * math.pi / 3.0
                verts[i * 6 + k, 0] = pos[i, 0] + math.cos(v_ang)
                verts[i * 6 + k, 1] = pos[i, 1] + math.sin(v_ang)
        
        centroid = np.mean(verts, axis=0)
        
        # Find best beta by trying 36 angles
        best_D = float('inf')
        best_beta = 0.0
        for bi in range(36):
            beta = bi * math.pi / 6.0
            D_val = 0.0
            for k in range(6):
                v_ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(v_ang)
                ny = math.sin(v_ang)
                projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                D_val = max(D_val, float(np.max(projs)), float(-np.min(projs)))
            if D_val < best_D:
                best_D = D_val
                best_beta = beta
        
        S = (best_D + APOTHEM) / APOTHEM
        return S, centroid, best_D, best_beta
    
    S_cur, cent_cur, D_cur, beta_cur = compute_S(positions, angles)
    S_best = S_cur
    pos_best = positions.copy()
    ang_best = angles.copy()
    cent_best = cent_cur.copy()
    beta_best = beta_cur
    
    # Simulated Annealing
    T_start = 0.3
    T_end = 0.0005
    n_iter = 30000
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    T = T_start
    
    t0 = time.time()
    for it in range(n_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        new_pos = positions.copy()
        new_ang = angles.copy()
        
        move_type = random.random()
        if move_type < 0.4:
            # Move one hexagon
            idx = random.randint(0, N - 1)
            new_pos[idx, 0] += random.gauss(0, 0.12)
            new_pos[idx, 1] += random.gauss(0, 0.12)
        elif move_type < 0.6:
            # Rotate one hexagon
            idx = random.randint(0, N - 1)
            new_ang[idx] += random.gauss(0, 15.0)
        elif move_type < 0.8:
            # Move two hexagons
            i1, i2 = random.sample(range(N), 2)
            new_pos[i1, 0] += random.gauss(0, 0.08)
            new_pos[i1, 1] += random.gauss(0, 0.08)
            new_pos[i2, 0] += random.gauss(0, 0.08)
            new_pos[i2, 1] += random.gauss(0, 0.08)
        else:
            # Global jitter
            new_pos[:, 0] += random.gauss(0, 0.02, size=N)
            new_pos[:, 1] += random.gauss(0, 0.02, size=N)
        
        S_new, cent_new, D_new, beta_new = compute_S(new_pos, new_ang)
        
        delta = S_new - S_cur
        if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
            positions = new_pos
            angles = new_ang
            S_cur = S_new
            cent_cur = cent_new
            beta_cur = beta_new
            
            if S_new < S_best:
                S_best = S_new
                pos_best = positions.copy()
                ang_best = angles.copy()
                cent_best = cent_cur.copy()
                beta_best = beta_new
        
        T *= cooling
    
    # Final refinement with scipy
    from scipy.optimize import minimize, minimize_scalar
    
    # Refine positions and angles
    def obj(params):
        pos = params[:2 * N].reshape(N, 2)
        ang = params[2 * N:]
        verts = np.zeros((N * 6, 2))
        for i in range(N):
            a = math.radians(ang[i])
            for k in range(6):
                v_ang = a + k * math.pi / 3.0
                verts[i * 6 + k, 0] = pos[i, 0] + math.cos(v_ang)
                verts[i * 6 + k, 1] = pos[i, 1] + math.sin(v_ang)
        centroid = np.mean(verts, axis=0)
        best_D = 0.0
        for bi in range(24):
            beta = bi * math.pi / 6.0
            for k in range(6):
                v_ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(v_ang)
                ny = math.sin(v_ang)
                projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                best_D = max(best_D, float(np.max(projs)), float(-np.min(projs)))
        return (best_D + APOTHEM) / APOTHEM
    
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            init_params = np.concatenate([pos_best.ravel(), ang_best])
            res = minimize(obj, init_params, method='Nelder-Mead',
                          options={'maxiter': 1000, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.fun < S_best:
                S_best = res.fun
                pos_best = res.x[:2 * N].reshape(N, 2)
                ang_best = res.x[2 * N:]
    except Exception:
        pass
    
    # Final beta refinement
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            verts = np.zeros((N * 6, 2))
            for i in range(N):
                a = math.radians(ang_best[i])
                for k in range(6):
                    v_ang = a + k * math.pi / 3.0
                    verts[i * 6 + k, 0] = pos_best[i, 0] + math.cos(v_ang)
                    verts[i * 6 + k, 1] = pos_best[i, 1] + math.sin(v_ang)
            centroid = np.mean(verts, axis=0)
            
            def beta_obj(b):
                D_val = 0.0
                for k in range(6):
                    v_ang = b + math.pi / 6.0 + k * math.pi / 3.0
                    nx = math.cos(v_ang)
                    ny = math.sin(v_ang)
                    projs = (verts[:, 0] - centroid[0]) * nx + (verts[:, 1] - centroid[1]) * ny
                    D_val = max(D_val, float(np.max(projs)), float(-np.min(projs)))
                return (D_val + APOTHEM) / APOTHEM
            
            res_b = minimize_scalar(beta_obj, bounds=(0, math.pi / 6), method='bounded',
                                   options={'maxiter': 300, 'xatol': 1e-11})
            if res_b.fun < S_best:
                S_best = res_b.fun
                beta_best = res_b.x
    except Exception:
        pass
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(pos_best[i, 0]), float(pos_best[i, 1]), float(ang_best[i])])
    
    center = [float(cent_best[0]), float(cent_best[1])]
    side = float(S_best)
    rotation = float(math.degrees(beta_best))
    
    return inner, center, side, rotation
