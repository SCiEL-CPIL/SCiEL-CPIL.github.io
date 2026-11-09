# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Strategy:
    1. Use the Icosahedron (12 vertices) as a strong geometric prior for the first 12 points.
    2. Initialize the remaining 2 points near the poles (0,0,±1) or other symmetric positions.
    3. Use L-BFGS-B for local refinement, which is more efficient than Nelder-Mead for smooth-ish objectives
       and handles the 42-dimensional space well.
    4. Perform a multi-start search with small random perturbations to escape local minima.
    5. The objective function is - (d_min / d_max)^2. We center the points to remove translational invariance
       and scale them to keep the optimizer in a reasonable range.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 590  # Leave 40s margin for safety
    
    # Objective function: Minimize negative squared ratio
    def objective(x_flat):
        # Check time
        if time.time() > deadline:
            return 1e10  # Large penalty if time runs out
        
        pts = x_flat.reshape(n, d)
        
        # Center of mass constraint: Subtract mean to handle translation invariance
        pts = pts - np.mean(pts, axis=0)
        
        # Calculate distances
        dists = pdist(pts)
        
        # Handle degenerate cases
        if np.any(np.isnan(dists)) or np.any(np.isinf(dists)):
            return 1e10
            
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        # We want to maximize ratio^2, so minimize -ratio^2
        return -ratio * ratio

    # Generate Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ])
    norms = np.linalg.norm(ico_verts, axis=1)
    ico_verts = ico_verts / norms[:, np.newaxis]
    
    # Base configuration: 12 Icosahedron points + 2 poles
    base_pts = np.zeros((n, d))
    base_pts[:12] = ico_verts
    base_pts[12] = [0, 0, 1]
    base_pts[13] = [0, 0, -1]
    
    # Normalize base configuration to have max distance 1
    dists_base = pdist(base_pts)
    d_max_base = np.max(dists_base)
    if d_max_base > 0:
        base_pts = base_pts / d_max_base
    
    best_score = -np.inf
    best_result = base_pts.flatten()
    
    # Multi-start L-BFGS-B optimization
    # We will try several random perturbations of the base configuration
    num_starts = 20
    rng = np.random.default_rng(42)
    
    for i in range(num_starts):
        if time.time() > deadline - 10:
            break
            
        # Create perturbed initial guess
        # Perturbation magnitude decreases with start index
        perturbation_scale = 0.1 * (1.0 - i / num_starts) + 0.01
        noise = rng.normal(0, perturbation_scale, (n, d))
        x0 = (base_pts + noise).flatten()
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < -1e9:  # Valid result
                if -res.fun > best_score:
                    best_score = -res.fun
                    best_result = res.x
                    
        except Exception as e:
            continue
            
    # Final Processing
    points = best_result.reshape(n, d)
    
    # Center of mass to 0
    points -= np.mean(points, axis=0)
    
    # Scale to unit max distance (optional, but keeps numbers nice)
    dists = pdist(points)
    d_max = np.max(dists)
    if d_max > 1e-12:
        points = points / d_max
        
    # Ensure finite
    if not np.all(np.isfinite(points)):
        # Ultimate fallback
        points = base_pts.copy()
        points -= np.mean(points, axis=0)
        dists = pdist(points)
        d_max = np.max(dists)
        if d_max > 0:
            points = points / d_max

    return points

# EVOLVE-BLOCK-END
