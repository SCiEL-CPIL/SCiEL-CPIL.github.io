import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) points.

    """
    n = 16
    
    # Objective function to minimize: -R = -(d_min / d_max)^2
    # We want to maximize R. Minimizing -R is equivalent.
    # To handle d_max = 0, we return a large positive value (bad for minimization).
    
    def objective(flat_points):
        pts = flat_points.reshape(n, 2)
        dists = pdist(pts)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max < 1e-12:
            return 1e10
        if d_min < 1e-12:
            return 1e10
            
        ratio = d_min / d_max
        return - (ratio ** 2)

    # Define bounds for the points.
    # Since the problem is scale-invariant and translation-invariant, we can fix the scale and center.
    # However, DE works best with explicit bounds.
    # Let's assume points are within [-1, 1] x [-1, 1] initially.
    # We can also fix one point to break translation symmetry and scale symmetry to reduce dimensionality,
    # but for 32 dims, DE can handle it. To be safe and fast, let's just bound them.
    
    # A better strategy for scale invariance: 
    # We can normalize the points inside the objective to d_max = 1? No, that makes the objective non-differentiable and potentially unstable for local optimizers.
    # Instead, we let the optimizer find the scale.
    # We must be careful with bounds. If points go too far, d_max increases, R decreases.
    # If points clump, d_min decreases, R decreases.
    # The optimal configuration will have d_max roughly constrained by the bounds if bounds are tight enough,
    # or the optimizer will naturally keep them compact.
    
    # Let's use bounds [-1, 1] for each coordinate.
    bounds = [(-1, 1)] * (2 * n)
    
    # Run Differential Evolution
    # maxiter=1000 might take a while. Let's estimate.
    # DE population size is typically 15 * dim = 15 * 32 = 480.
    # 1000 iterations * 480 evaluations * cost of objective.
    # Cost of objective: pdist on 16 points is fast (~120 dists).
    # 480,000 evaluations. Should be well within 600s.
    
    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=500, 
            popsize=20, 
            tol=1e-10, 
            seed=42,
            polish=True,  # This runs L-BFGS-B on the best result
            init='latinhypercube'
        )
        
        best_points = result.x.reshape(n, 2)
        
    except Exception:
        # Fallback to a simple grid if DE fails or times out (unlikely to time out in 500 iters)
        points = np.zeros((n, 2))
        idx = 0
        for j in range(4):
            for i in range(4):
                points[idx, 0] = i + 0.5 * (j % 2)
                points[idx, 1] = j * np.sqrt(3) / 2.0
                idx += 1
        points -= np.mean(points, axis=0)
        # Scale to fit in [-1,1]
        d_max_init = np.max(pdist(points))
        points /= d_max_init
        best_points = points

    # Additional local refinement using Powell's method which is robust for non-smooth functions
    # DE with polish=True already does L-BFGS-B, which assumes smoothness. 
    # The min/max function is not smooth. Powell is better for non-smooth local refinement.
    
    result_powell = minimize(
        objective, 
        best_points.flatten(), 
        method='Powell', 
        options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
    )
    
    best_points = result_powell.x.reshape(n, 2)
    
    # One more pass with Nelder-Mead for final polish if needed, or just return Powell result.
    # Powell is generally sufficient.
    
    # Verify shape and finiteness
    if not np.all(np.isfinite(best_points)):
        # Fallback
        points = np.zeros((n, 2))
        idx = 0
        for j in range(4):
            for i in range(4):
                points[idx, 0] = i + 0.5 * (j % 2)
                points[idx, 1] = j * np.sqrt(3) / 2.0
                idx += 1
        points -= np.mean(points, axis=0)
        best_points = points

    return best_points
