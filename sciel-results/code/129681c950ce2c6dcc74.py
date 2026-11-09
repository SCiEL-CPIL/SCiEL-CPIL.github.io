import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 11
H = np.sqrt(3) / 2

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

def is_inside(points):
    """
    Check if all points are inside the equilateral triangle (0,0), (1,0), (0.5, H).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Check y >= 0
    if np.any(y < -1e-9):
        return False
    
    # Check y <= sqrt(3) * x
    if np.any(y > np.sqrt(3) * x + 1e-9):
        return False
        
    # Check y <= sqrt(3) * (1 - x)
    if np.any(y > np.sqrt(3) * (1 - x) + 1e-9):
        return False
        
    return True

def objective(params):
    """
    Objective function for minimization.
    Returns -min_area if valid, penalty if invalid.
    """
    points = params.reshape(N, 2)
    
    if not is_inside(points):
        return 1.0  # Penalty: larger than any possible area (max area ~ 0.04)
        
    area = min_triangle_area_vec(points)
    return -area

def construct():
    start_time = time.time()
    timeout = 130  # Safety margin
    
    # Bounds for Cartesian coordinates
    # x in [0, 1], y in [0, H]
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    best_points = None
    best_score = -float('inf')
    
    # Use Differential Evolution for global search
    try:
        # maxiter and popsize can be tuned based on time
        # 150s budget allows for thorough search
        result = differential_evolution(
            objective,
            bounds,
            maxiter=500,
            popsize=15,
            tol=0.01,
            seed=42,
            polish=True,  # Use Nelder-Mead to polish the best solution
            init='sobol' # Better coverage of the search space
        )
        
        best_params = result.x
        best_points = best_params.reshape(N, 2)
        
        if is_inside(best_points):
            best_score = min_triangle_area_vec(best_points)
        else:
            # Fallback if DE returned an invalid point (should not happen with penalty)
            best_points = None
            
    except Exception:
        pass
        
    # If DE didn't find a valid point or failed, try a few random restarts with local optimization
    if best_points is None:
         # Fallback to a simple valid configuration if time is tight or failure
         # Generate a valid random point set
         while True:
             # Sample in bounding box
             pts = np.random.rand(N, 2)
             pts[:, 0] *= 1.0
             pts[:, 1] *= H
             if is_inside(pts):
                 best_points = pts
                 break
         return best_points

    # Check time again before returning
    if time.time() - start_time > timeout:
        return best_points

    # Local refinement using Nelder-Mead
    def refine_local(points, start_time, deadline):
        if time.time() > deadline:
            return points
        
        best_points = points.copy()
        best_score = min_triangle_area_vec(best_points)
        
        # Try a few local optimizations with different initial perturbations
        # This helps escape shallow local minima found by DE
        rng = np.random.default_rng(42)
        
        for attempt in range(5):
            if time.time() > deadline:
                break
                
            # Create a slightly perturbed version to start from
            if attempt == 0:
                x0 = best_points.flatten()
            else:
                x0 = best_points.flatten() + rng.normal(0, 0.01, size=2*N)
                # Project back inside if needed (simple clip)
                x0 = np.clip(x0, 0, 1)
                x0[1::2] = np.clip(x0[1::2], 0, H)
            
            def obj_local(params):
                pts = params.reshape(N, 2)
                if not is_inside(pts):
                    return 1.0
                return -min_triangle_area_vec(pts)
            
            res = minimize(obj_local, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.x is not None:
                pts = res.x.reshape(N, 2)
                if is_inside(pts):
                    score = min_triangle_area_vec(pts)
                    if score > best_score:
                        best_score = score
                        best_points = pts
        
        return best_points

    best_points = refine_local(best_points, start_time, timeout)
    
    # Check time again before returning
    if time.time() - start_time > timeout:
        return best_points

    return best_points
