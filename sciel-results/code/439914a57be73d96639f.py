N = 11

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

D_MIN = np.sqrt(3)
# Normals for a hexagon with apothem R.
# The constraint is dot(p, n) <= R.
# For a regular hexagon, the 6 normals are spaced by 60 degrees.
# Let's align one normal with the x-axis.
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
normals = np.column_stack((np.cos(angles), np.sin(angles)))

def compute_R(points):
    # points: (N, 2)
    # Returns the minimum apothem R required to contain all points
    # in a hexagon centered at origin with fixed orientation.
    proj = points @ normals.T
    return np.max(proj)

def objective(x):
    points = x.reshape(N, 2)
    R = compute_R(points)
    
    diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    
    if min_dist < D_MIN:
        penalty = (D_MIN - min_dist) ** 2
    else:
        penalty = 0.0
        
    return R + 10.0 * penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_R = float('inf')
    best_points = None
    
    # Initial configurations
    inits = []
    
    # 1. 7-hexagon flower + 4 random
    flower = np.zeros((7, 2))
    for i in range(6):
        angle = i * np.pi / 3
        flower[i+1] = [D_MIN * np.cos(angle), D_MIN * np.sin(angle)]
    
    # Add 4 more points in a ring further out
    for i in range(4):
        angle = (i + 0.5) * np.pi / 2 # 45, 135, 225, 315
        dist = D_MIN * 1.8
        inits.append(np.vstack([flower, np.array([dist * np.cos(angle), dist * np.sin(angle)])]))

    # 2. Random starts
    for _ in range(15):
        pts = np.random.uniform(-3, 3, (N, 2))
        inits.append(pts)

    # Optimization loop
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = minimize(objective, init.flatten(), method='Nelder-Mead', 
                           options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
            # Check validity
            pts = res.x.reshape(N, 2)
            diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist >= D_MIN - 1e-6:
                # Recompute R to be sure
                R_val = compute_R(pts)
                if R_val < best_R:
                    best_R = R_val
                    best_points = pts
        except Exception:
            pass

    # If no valid points found, return a safe default
    if best_points is None:
        inner = []
        for k in range(N):
            inner.append([k * D_MIN, 0.0, 0.0])
        return inner, (0.0, 0.0), 20.0, 0.0

    # Calculate side length L from best_R
    L = best_R * 2.0 / D_MIN + 1.0
    
    inner = []
    for i in range(N):
        inner.append([best_points[i, 0], best_points[i, 1], 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
