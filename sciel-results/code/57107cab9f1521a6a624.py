# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize

def _score(points):
    """Compute R = (d_min / d_max)^2."""
    n = points.shape[0]
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max < 1e-15:
        return 0.0
    return (d_min / d_max) ** 2

def _neg_soft_bottleneck(x, beta, max_dist=1.0):
    """
    Objective function for the soft bottleneck optimization.
    We want to maximize d_min subject to d_i <= max_dist for all i.
    
    Strategy:
    1. Soft min of distances: approximates d_min. We want to maximize this.
    2. Penalty for distances > max_dist: We want to minimize this.
    
    Total objective to MINIMIZE:
    -soft_min + lambda * soft_max_exceedance
    
    However, a simpler and often more effective approach for packing is:
    Minimize: -soft_min(dists) + penalty * sum(max(0, dists - max_dist)^2)
    
    But since we can normalize the output at the end, we can just fix the scale
    during optimization. Let's fix the bounding box or max distance.
    
    Actually, a robust method for "max-min distance" is to minimize the sum of 
    inverse distances or similar, but the constraint d_i <= 1 is key.
    
    Let's use a different formulation:
    Minimize: - (1/beta) * log( sum(exp(-beta * d_i)) )  [This is -soft_min]
    Subject to: d_i <= 1.
    
    To handle the constraint in a gradient-based optimizer, we add a penalty term:
    Penalty = sum( max(0, d_i - 1.0)^2 )
    
    So we minimize: -soft_min + lambda * penalty
    
    As we iterate, we can increase lambda to enforce the constraint strictly.
    """
    pts = x.reshape(16, 2)
    dists = spd.pdist(pts)
    if len(dists) == 0:
        return 0.0
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    neg_d = -beta * dists
    max_neg_d = np.max(neg_d)
    log_sum_exp_neg = max_neg_d + np.log(np.sum(np.exp(neg_d - max_neg_d)))
    soft_min = -log_sum_exp_neg / beta
    
    # We want to MAXIMIZE soft_min, so we MINIMIZE -soft_min
    obj = -soft_min
    
    # Penalty for violating d_i <= 1.0
    # Using a smooth hinge loss
    exceed = np.maximum(0.0, dists - max_dist)
    penalty = np.sum(exceed ** 2)
    
    # Lambda is passed via closure or global? Let's use a fixed high lambda for the soft phase
    # Or we can just return the penalty if any violation is large, but smooth is better for gradients
    lambda_pen = 10.0
    
    return obj + lambda_pen * penalty

def _refine_exact(points):
    """
    Refine points using exact objective with L-BFGS-B.
    Objective: Minimize - (d_min / d_max)^2
    Since this is non-smooth, we use a small perturbation or just rely on the 
    fact that L-BFGS-B with finite differences can often navigate the flat 
    regions if initialized well.
    
    Actually, a better exact objective for the final step is to minimize 
    -d_min + penalty * max(0, d_max - 1). But we normalized d_max to 1.
    
    Let's stick to the ratio, but use a "softened" version of the min/max 
    with a very high beta to make it nearly exact but differentiable.
    """
    def exact_obj(x):
        pts = x.reshape(16, 2)
        dists = spd.pdist(pts)
        if len(dists) == 0:
            return 0.0
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max < 1e-15:
            return 0.0
        return -(d_min / d_max) ** 2
    
    # L-BFGS-B is good for box constraints, but we have no box constraints here 
    # except implicit finite values. 
    # We will center and normalize after each step? No, that breaks the gradient.
    # We just minimize in the current space.
    
    res = minimize(exact_obj, points.flatten(), method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    
    if res.success or res.fun < exact_obj(points.flatten()):
        best_pts = res.x.reshape(16, 2)
        # Normalize
        best_pts = best_pts - best_pts.mean(axis=0)
        norm = np.max(np.linalg.norm(best_pts, axis=1))
        if norm > 1e-10:
            best_pts = best_pts / norm
        return best_pts
    return points

def _generate_hexagonal_16():
    """Generate a hexagonal lattice pattern for 16 points."""
    # A hexagonal packing of 19 points has 1 in center, 6 in ring 1, 12 in ring 2.
    # We need 16. We can take the center + 6 ring 1 + 9 from ring 2?
    # Or just a hex grid and pick the best 16.
    # Let's generate a hex grid of radius 2 (19 points) and remove the 3 worst (outermost)
    # or just use a known good config.
    
    # Hex grid points
    pts = []
    a = 1.0 # spacing
    for q in range(-2, 3):
        for r in range(-2, 3):
            # Hex to Cartesian
            x = a * (q + r/2.0)
            y = a * (np.sqrt(3)/2.0 * r)
            pts.append([x, y])
    pts = np.array(pts)
    
    # We have 25 points. We need 16.
    # The center is (0,0). The 6 neighbors are distance 1.
    # The 12 next neighbors are distance sqrt(3) ~ 1.732.
    # If we take center + 6 neighbors + 9 of the 12 outer, we get 16.
    # Which 9 to take? Any 9 will be symmetric enough.
    
    # Sort by distance from center
    dists_from_center = np.linalg.norm(pts, axis=1)
    sorted_indices = np.argsort(dists_from_center)
    
    # Take first 16
    # But this might be a cluster. Let's check the spread.
    # Center (0), 6 at d=1, 12 at d=sqrt(3).
    # Taking 1+6+9 = 16 points.
    # This forms a hexagon with one corner missing? No, it's 3 corners missing.
    # This should be a good starting point.
    
    selected = pts[sorted_indices[:16]]
    
    # Center and normalize
    selected = selected - selected.mean(axis=0)
    norm = np.max(np.linalg.norm(selected, axis=1))
    if norm > 1e-10:
        selected = selected / norm
    return selected

def _generate_random_16():
    """Generate random points in a unit square, then normalize."""
    pts = np.random.rand(16, 2) * 2 - 1 # [-1, 1]
    pts = pts - pts.mean(axis=0)
    norm = np.max(np.linalg.norm(pts, axis=1))
    if norm > 1e-10:
        pts = pts / norm
    return pts

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_points = None
    best_score = -1.0
    
    # Initializations
    inits = []
    inits.append(_generate_hexagonal_16())
    
    # Add a few random inits
    for _ in range(5):
        inits.append(_generate_random_16())
    
    # Optimization Strategy:
    # 1. Soft Bottleneck with L-BFGS-B (fast, gradient-based)
    # 2. Refine with Exact Objective
    
    for i_init, init_pts in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        current_pts = init_pts.copy()
        
        # Phase 1: Soft optimization
        # We use a moderate beta. 
        # The objective is -soft_min + penalty.
        # L-BFGS-B is efficient for this.
        
        try:
            # Run L-BFGS-B on the soft objective
            # We don't have bounds, but the penalty keeps points somewhat close.
            # To prevent explosion, we can add a small regularization to keep points bounded.
            # Or just rely on the penalty.
            
            def soft_obj_with_reg(x):
                pts = x.reshape(16, 2)
                # Regularization to keep points from flying away
                reg = 0.01 * np.sum(pts**2)
                return _neg_soft_bottleneck(x, beta=20.0) + reg
            
            res = minimize(
                soft_obj_with_reg, 
                current_pts.flatten(), 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            
            if res.success or res.fun < soft_obj_with_reg(current_pts.flatten()):
                current_pts = res.x.reshape(16, 2)
                
                # Normalize
                current_pts = current_pts - current_pts.mean(axis=0)
                norm = np.max(np.linalg.norm(current_pts, axis=1))
                if norm > 1e-10:
                    current_pts = current_pts / norm
                    
        except Exception:
            pass
        
        # Phase 2: Exact Refinement
        # Now that we are in a good basin, use the exact metric.
        try:
            refined_pts = _refine_exact(current_pts)
            current_pts = refined_pts
        except Exception:
            pass
        
        # Evaluate
        score = _score(current_pts)
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
        # If we have time, do a second round of soft optimization with higher beta?
        # Or just move to next init.
        
    # If best_points is still None (shouldn't happen), fallback
    if best_points is None:
        best_points = _generate_hexagonal_16()
        
    # Final check and normalization
    best_points = best_points - best_points.mean(axis=0)
    norm = np.max(np.linalg.norm(best_points, axis=1))
    if norm > 1e-10:
        best_points = best_points / norm
        
    return best_points

# EVOLVE-BLOCK-END
