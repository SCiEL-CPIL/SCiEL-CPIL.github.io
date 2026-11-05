import numpy as np
import time
from scipy.optimize import minimize

N = 11
START_TIME = time.time()
DEADLINE = 150.0

def construct():
    best_S = 40.0
    best_xy = None
    
    angles = np.deg2rad(np.arange(30, 360, 60))
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    min_dist = np.sqrt(3.0)
    
    def objective(params):
        S = params[-1]
        xy = params[:-1].reshape(N, 2)
        
        # Boundary violation
        A = S * np.sqrt(3) / 2
        dists = np.abs(xy @ normals.T)
        boundary_viol = np.sum(np.maximum(0, dists - A) ** 2)
        
        # Overlap violation
        diff = xy[:, np.newaxis, :] - xy[np.newaxis, :, :]
        dists_sq = np.sum(diff ** 2, axis=2)
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        dists = np.sqrt(dists_sq[mask])
        overlap_viol = np.sum(np.maximum(0, min_dist - dists) ** 2)
        
        penalty = 500.0
        return S + penalty * (boundary_viol + overlap_viol)
        
    # Initializations
    inits = []
    
    # 1. Random
    for i in range(5):
        S_init = 4.0
        xs = np.random.uniform(-S_init, S_init, N)
        ys = np.random.uniform(-S_init, S_init, N)
        inits.append(np.concatenate([xs, ys, [S_init]]))
        
    # 2. 7-hex cluster + 4 random
    # 7-hex cluster centers
    c0 = [0,0]
    c1 = [np.sqrt(3), 0]
    c2 = [-np.sqrt(3), 0]
    c3 = [np.sqrt(3)/2, 1.5]
    c4 = [-np.sqrt(3)/2, 1.5]
    c5 = [np.sqrt(3)/2, -1.5]
    c6 = [-np.sqrt(3)/2, -1.5]
    base = np.array(c0+c1+c2+c3+c4+c5+c6)
    for i in range(5):
        # Add 4 random points near the boundary
        extra_r = np.random.uniform(1.5, 2.5, 4)
        extra_t = np.random.uniform(0, 2*np.pi, 4)
        ex = extra_r * np.cos(extra_t)
        ey = extra_r * np.sin(extra_t)
        xy_init = np.vstack([base, np.column_stack((ex, ey))])
        S_init = 4.0
        inits.append(np.concatenate([xy_init.flatten(), [S_init]]))
        
    for init in inits:
        if time.time() > DEADLINE - 10:
            break
        try:
            res = minimize(objective, init, method='BFGS', options={'maxiter': 300, 'gtol': 1e-6})
            if res.fun < best_S:
                S_opt = res.x[-1]
                xy_opt = res.x[:-1].reshape(N, 2)
                
                # Check feasibility strictly
                A = S_opt * np.sqrt(3) / 2
                dists = np.abs(xy_opt @ normals.T)
                if np.all(dists <= A + 1e-5):
                    diff = xy_opt[:, np.newaxis, :] - xy_opt[np.newaxis, :, :]
                    dists_sq = np.sum(diff ** 2, axis=2)
                    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
                    dists = np.sqrt(dists_sq[mask])
                    if np.all(dists >= min_dist - 1e-5):
                        best_S = S_opt
                        best_xy = xy_opt
        except:
            pass
            
    if best_xy is not None:
        inner = [[best_xy[i][0], best_xy[i][1], 0.0] for i in range(N)]
        return inner, (0.0, 0.0), best_S, 0.0
    else:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
