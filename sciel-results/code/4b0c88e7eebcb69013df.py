import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
A = np.array([0.0, 0.0])
B = np.array([1.0, 0.0])
C = np.array([0.5, H])

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def params_to_points(params):
    """
    Maps a flat array of 2*N parameters (u_i, v_i) to points in the triangle.
    Uses normalization to ensure u+v <= 1.
    """
    params = np.array(params).reshape(N, 2)
    u = params[:, 0]
    v = params[:, 1]
    
    # If u+v > 1, normalize to lie on the edge u+v=1 (mapping square to triangle)
    # This ensures points stay inside the triangle.
    s = u + v
    mask = s > 1.0
    u[mask] = u[mask] / s[mask]
    v[mask] = v[mask] / s[mask]
    
    # Points = A + u*(B-A) + v*(C-A)
    # B-A = (1, 0)
    # C-A = (0.5, H)
    x = u * 1.0 + v * 0.5
    y = u * 0.0 + v * H
    
    return np.column_stack((x, y))

def objective(params):
    """
    Objective function to minimize (negative of min triangle area).
    """
    points = params_to_points(params)
    return -min_triangle_area_vec(points)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Initial random start
    rng = np.random.default_rng(42)
    
    # Bounds for DE: u, v in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_params = None
    best_score = -np.inf
    
    # Phase 1: Global Search with Differential Evolution
    # We limit iterations based on time, but DE has a fixed maxiter.
    # We can run multiple shorter DE runs or one longer one.
    # Let's try one run with maxiter=100, popsize=15.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            seed=42, 
            tol=1e-6,
            polish=False # We will do our own polish
        )
        best_params = result.x
        best_score = -result.fun
        
        # Check time
        if time.time() > deadline - 30.0:
             # If we are running out of time, skip local refinement or do minimal
             pass
             
    except Exception as e:
        print(f"DE failed: {e}")
        # Fallback to random
        best_params = rng.random(2 * N)
        best_score = -objective(best_params)

    # Phase 2: Local Refinement (Polish)
    # Use multiple restarts of local optimization from the best DE result
    # and some random points near it.
    
    if time.time() < deadline - 10.0:
        current_params = best_params.copy()
        current_score = best_score
        
        # Simple local search: try small perturbations
        # Or use scipy minimize with L-BFGS-B on a smoothed objective?
        # The objective is non-smooth. L-BFGS-B might struggle.
        # Let's use a custom simple hill climbing / random restart local search.
        
        local_iters = 0
        max_local_iters = 5000
        step_size = 0.05
        
        while time.time() < deadline - 5.0 and local_iters < max_local_iters:
            local_iters += 1
            
            # Generate a perturbed version
            # Gaussian noise
            noise = rng.normal(0, step_size, size=(2 * N,))
            new_params = current_params + noise
            
            # Clip to bounds [0, 1]
            new_params = np.clip(new_params, 0.0, 1.0)
            
            new_score = -objective(new_params)
            
            if new_score > current_score:
                current_score = new_score
                current_params = new_params
                # Reduce step size slightly if we improve? Or keep constant.
                # Let's keep constant for simplicity, or decay slowly.
                step_size = max(0.001, step_size * 0.99)
            else:
                # If no improvement for a while, maybe restart from best_params?
                # For now, just continue.
                pass
                
        if current_score > best_score:
            best_score = current_score
            best_params = current_params

    # Convert best parameters to points
    points = params_to_points(best_params)
    
    # Final sanity check: ensure points are within triangle (should be by construction)
    # Return the points
    return points
