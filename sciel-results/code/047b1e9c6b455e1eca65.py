import numpy as np
import time
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initial configuration: 3x4 grid of hexagons
    # Flat-top hexagons, side length 1
    # Horizontal spacing: 1.5
    # Vertical spacing: sqrt(3)/2
    # Odd row offset: 0.75
    
    sqrt3 = np.sqrt(3)
    
    centers = []
    for r in range(3):
        for c in range(4):
            x = c * 1.5
            if r % 2 == 1:
                x += 0.75
            y = r * (sqrt3 / 2)
            centers.append([x, y])
    
    centers = np.array(centers)
    
    # Center the cluster
    centroid = np.mean(centers, axis=0)
    centers = centers - centroid
    
    # Initial outer hexagon parameters
    # c_x, c_y, theta
    init_c = np.array([0.0, 0.0])
    init_theta = 0.0
    
    # Parameters vector: [x0, y0, ..., x11, y11, cx, cy, theta]
    x0 = np.concatenate([centers.flatten(), init_c, [init_theta]])
    
    def objective(params):
        if time.time() > deadline:
            return 1e10
        
        pts = params[:2*N].reshape(N, 2)
        cx = params[2*N]
        cy = params[2*N + 1]
        theta = params[2*N + 2]
        
        # Check disjointness
        # Distance between centers must be >= sqrt(3)
        # Compute pairwise distances
        # Efficient way:
        dx = pts[:, 0, None] - pts[:, 0, None].T
        dy = pts[:, 1, None] - pts[:, 1, None].T
        dists = np.sqrt(dx**2 + dy**2)
        # Mask diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < sqrt3:
            violation = (sqrt3 - min_dist)**2
        else:
            violation = 0.0
        
        # Compute minimal enclosing hexagon side length
        # Normals for regular hexagon with rotation theta
        # Side normals are at angles theta + k * pi/3
        # Apothem h = S * sqrt(3) / 2
        # S = 2 * h / sqrt(3)
        # h = max over points and normals of |dot(p - c, n)|
        
        angles = theta + np.arange(6) * (np.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # p - c
        rel_pts = pts - np.array([cx, cy])
        
        # Project points onto normals
        # rel_pts (N, 2) @ normals.T (2, 6) -> (N, 6)
        projections = rel_pts @ normals.T
        
        # Absolute max projection
        max_proj = np.max(np.abs(projections))
        
        h = max_proj
        S = 2 * h / sqrt3
        
        # Objective: Minimize S, penalize violations
        return S + 100.0 * violation
    
    # Run optimization
    # Use Nelder-Mead for robustness with small dimensions
    # Maxiter and maxfev to stay within time limit
    # 27 dimensions.
    
    best_result = None
    best_score = np.inf
    
    try:
        # Multiple restarts might help, but let's start with one
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
        
        if res.fun < best_score:
            best_score = res.fun
            best_result = res.x
            
    except Exception:
        pass
    
    # If time permits, do a second run with different initial theta
    if time.time() < deadline - 10:
        try:
            x0_2 = x0.copy()
            x0_2[2*N+2] = np.pi/6  # Rotate outer hex
            res2 = minimize(objective, x0_2, method='Nelder-Mead', 
                       options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
            if res2.fun < best_score:
                best_score = res2.fun
                best_result = res2.x
        except Exception:
            pass
            
    if best_result is None:
        # Fallback to initial
        best_result = x0
        best_score = objective(x0)
        
    # Extract results
    pts = best_result[:2*N].reshape(N, 2)
    cx = best_result[2*N]
    cy = best_result[2*N + 1]
    theta = best_result[2*N + 2]
    
    # Calculate final side length
    angles = theta + np.arange(6) * (np.pi / 3)
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    rel_pts = pts - np.array([cx, cy])
    projections = rel_pts @ normals.T
    max_proj = np.max(np.abs(projections))
    S = 2 * max_proj / sqrt3
    
    # Construct inner list
    # Inner hexagons have angle 0
    inner = [[float(pts[i][0]), float(pts[i][1]), 0.0] for i in range(N)]
    
    # Return
    return inner, (float(cx), float(cy)), float(S), float(theta)

