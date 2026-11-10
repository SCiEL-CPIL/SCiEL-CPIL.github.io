import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.distance import pdist, squareform

N = 100

def construct():
    # 1. Generate FCC lattice points
    # We generate a large block and pick the 100 closest to the center
    # FCC condition: i + j + k is even
    range_val = 8
    points = []
    for i in range(-range_val, range_val + 1):
        for j in range(-range_val, range_val + 1):
            for k in range(-range_val, range_val + 1):
                if (i + j + k) % 2 == 0:
                    points.append([i, j, k])
    
    points = np.array(points, dtype=float)
    
    # Center the points
    center = np.mean(points, axis=0)
    points -= center
    
    # Sort by distance to origin and take the first N
    dists = np.sum(points**2, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:N]]
    
    # Recenter to ensure symmetry
    center = np.mean(points, axis=0)
    points -= center

    # 2. Calculate D_min (minimum pairwise distance)
    # Use pdist for efficiency and robustness
    dist_matrix = pdist(points)
    D_min = np.min(dist_matrix)
    
    # 3. Define objective function
    # We optimize the rotation angles (roll, pitch, yaw)
    # Objective: Maximize the minimum clearance? No, we want to minimize W_max 
    # for a fixed D_min, then scale. 
    # The radius r is determined by s * D_min / 2 = 0.5 - s * W_max
    # s = 0.5 / (W_max + D_min/2)
    # r = s * D_min / 2 = 0.25 * D_min / (W_max + D_min/2)
    # To maximize r, we need to minimize W_max.
    
    def rot_matrix(angles):
        roll, pitch, yaw = angles
        c_r, s_r = np.cos(roll), np.sin(roll)
        c_p, s_p = np.cos(pitch), np.sin(pitch)
        c_y, s_y = np.cos(yaw), np.sin(yaw)
        
        Rx = np.array([
            [1, 0, 0],
            [0, c_r, -s_r],
            [0, s_r, c_r]
        ])
        Ry = np.array([
            [c_p, 0, s_p],
            [0, 1, 0],
            [-s_p, 0, c_p]
        ])
        Rz = np.array([
            [c_y, -s_y, 0],
            [s_y, c_y, 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def objective(angles):
        R = rot_matrix(angles)
        rotated = points @ R.T
        W_max = np.max(np.abs(rotated))
        return W_max

    # 4. Optimize
    # Use Differential Evolution for global search
    # Bounds for angles: -pi to pi
    bounds = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    
    # Run DE with a reasonable population size
    de_res = differential_evolution(objective, bounds, maxiter=100, tol=1e-8, seed=42, polish=False)
    
    best_angles = de_res.x
    best_w_max = de_res.fun
    
    # Refine with Nelder-Mead
    res = minimize(objective, best_angles, method='Nelder-Mead', options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12})
    if res.fun < best_w_max:
        best_w_max = res.fun
        best_angles = res.x

    R_opt = rot_matrix(best_angles)
    
    # 5. Calculate scale and final positions
    s = 0.5 / (best_w_max + D_min / 2.0)
    centers = s * (points @ R_opt.T)
    
    return centers
