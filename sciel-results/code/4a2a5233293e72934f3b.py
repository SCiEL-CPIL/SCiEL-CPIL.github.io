"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist

def _compute_score(points):
    """Compute the squared ratio of max to min pairwise distance."""
    n = points.shape[0]
    # Compute all pairwise distances
    dists = pdist(points)
    if len(dists) == 0:
        return 1e10
    min_d = np.min(dists)
    max_d = np.max(dists)
    if min_d < 1e-10:
        return 1e10
    return (max_d / min_d) ** 2

def construct():
    n = N
    d = D
    
    # Objective function for direct coordinate optimization
    # We use a smooth approximation of the ratio of max to min squared distance.
    # min(d_ij^2) and max(d_ij^2) are non-smooth.
    # We approximate max(x) with T * log(sum(exp(x/T))) and min(x) with -T * log(sum(exp(-x/T))).
    
    def objective_coords(x):
        points = x.reshape(n, d)
        # Compute squared distances
        d_sq = pdist(points) ** 2
        
        # Check for degeneracy
        if np.any(d_sq < 1e-12):
            return 1e10
            
        # Smooth max and min
        T = 0.5
        # Numerical stability: subtract max before exp
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        # Approximation of max(d_sq)
        # exp(d_sq / T) can overflow if d_sq is large.
        # Use log-sum-exp trick
        max_approx = T * np.log(np.sum(np.exp((d_sq - max_d_sq) / T))) + max_d_sq
        
        # Approximation of min(d_sq)
        min_approx = -T * np.log(np.sum(np.exp((min_d_sq - d_sq) / T))) + min_d_sq
        
        if min_approx < 1e-12:
            return 1e10
            
        # The objective is (max_d / min_d)^2 = max_d_sq / min_d_sq
        ratio = max_approx / min_approx
        
        # Add a small regularization to prevent points from collapsing or exploding
        # This helps with scale invariance and numerical stability
        # We want the points to be centered and have a reasonable scale
        center = np.mean(points, axis=0)
        centered = points - center
        trace = np.sum(centered ** 2)
        # Penalize if the total variance is too far from 1
        scale_penalty = (trace - n) ** 2
        
        return ratio + 0.1 * scale_penalty

    # Initialization
    np.random.seed(42)
    
    # Strategy: Use multiple random initializations and pick the best one.
    # For n=14 in d=3, random starts can be trapped in local minima.
    # A better start is often a regular simplex or a lattice, but random with good optimization is robust.
    
    best_points = None
    best_score = 1e10
    
    num_starts = 10
    
    for i in range(num_starts):
        # Random Gaussian initialization
        x0 = np.random.randn(n, d)
        
        # Optional: Center and scale
        x0 = x0 - np.mean(x0, axis=0)
        
        try:
            res = minimize(objective_coords, x0.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
            
            if res.success or res.fun < best_score:
                points = res.x.reshape(n, d)
                score = _compute_score(points)
                
                if score < best_score:
                    best_score = score
                    best_points = points
                    
        except Exception:
            continue
            
    if best_points is None:
        # Fallback
        return np.random.randn(n, d)
        
    # Final refinement with a smaller temperature T for sharper max/min approximation
    def objective_final(x):
        points = x.reshape(n, d)
        d_sq = pdist(points) ** 2
        if np.any(d_sq < 1e-12):
            return 1e10
            
        T = 0.1
        max_d_sq = np.max(d_sq)
        min_d_sq = np.min(d_sq)
        
        max_approx = T * np.log(np.sum(np.exp((d_sq - max_d_sq) / T))) + max_d_sq
        min_approx = -T * np.log(np.sum(np.exp((min_d_sq - d_sq) / T))) + min_d_sq
        
        if min_approx < 1e-12:
            return 1e10
            
        ratio = max_approx / min_approx
        
        center = np.mean(points, axis=0)
        centered = points - center
        trace = np.sum(centered ** 2)
        scale_penalty = (trace - n) ** 2
        
        return ratio + 0.01 * scale_penalty

    try:
        res_final = minimize(objective_final, best_points.flatten(), method='L-BFGS-B', 
                             options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if res_final.fun < best_score:
            final_points = res_final.x.reshape(n, d)
            final_score = _compute_score(final_points)
            if final_score < best_score:
                best_points = final_points
                best_score = final_score
    except Exception:
        pass

    return best_points

# EVOLVE-BLOCK-END
