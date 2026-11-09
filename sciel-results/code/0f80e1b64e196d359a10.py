# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist
import itertools


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    
    # We fix one point at the origin to break translational symmetry.
    # The remaining 15 points are free.
    # We also normalize such that the maximum distance is 1 (or close to it) to enforce scale invariance.
    # However, optimizing directly on the ratio R = (d_min/d_max)^2 is non-smooth due to min/max.
    # Strategy:
    # 1. Represent points as complex numbers z_1, ..., z_16.
    # 2. Fix z_0 = 0.
    # 3. Let z_1, ..., z_15 be variables.
    # 4. Objective: Maximize min_{i<j} |z_i - z_j| / max_{i<j} |z_i - z_j|.
    #    Let f(x) = min_pair_dist / max_pair_dist.
    #    We want to maximize f(x)^2.
    
    # A common trick to make this smooth is to optimize the log of the minimum distance
    # using a soft-min (log-sum-exp) approximation, and constrain the maximum distance to be 1.
    # But constraint optimization with L-BFGS-B is tricky if the constraint is active.
    
    # Alternative approach:
    # Maximize d_min subject to d_max <= 1.
    # Then the ratio is d_min / 1 = d_min.
    # Since the problem is scale invariant, fixing d_max=1 is valid.
    # However, d_min is not differentiable.
    
    # Let's use the smooth surrogate for the minimum distance:
    # soft_min(d) = -1/t * log( sum(exp(-t*d_i)) )
    # As t -> infinity, soft_min -> min(d_i).
    # We want to maximize soft_min(d) subject to max(d) <= 1.
    
    # Actually, let's just maximize the ratio directly using a smooth approximation.
    # Let D be the vector of pairwise distances.
    # R = (min(D) / max(D))^2.
    # log(R) = 2 * (log(min(D)) - log(max(D))).
    # log(min(D)) can be approximated by soft_min_log.
    # log(max(D)) can be approximated by soft_max_log.
    
    # Let's try a simpler heuristic first: Start with a good initial configuration.
    # For 16 points, a hexagonal lattice or a circle might be good starts.
    # But we need to optimize.
    
    # Let's define the variables as the real and imaginary parts of z_1...z_15.
    # z_0 = 0.
    
    # We will maximize the soft-min of distances, while penalizing if max distance > 1.
    # Or, we can fix the scale by normalizing the points after each step? No, that breaks gradient.
    
    # Better: Use the fact that if we scale all points by 1/d_max, d_max becomes 1 and d_min becomes d_min/d_max.
    # So maximizing d_min/d_max is equivalent to maximizing d_min subject to d_max=1.
    
    # Let's use a penalty method for the constraint d_max <= 1.
    # Objective: maximize soft_min(D) - lambda * max(0, max(D) - 1)^2
    
    # Let's try a different, more robust approach:
    # Random restarts with L-BFGS-B on the smooth surrogate.
    
    def compute_distances(points):
        # points is (16, 2)
        return pdist(points)
    
    def objective(x, t=50.0):
        # x is (30,) -> 15 points * 2 coords
        # z_0 = 0
        z = np.zeros((n, 2))
        z[1:, 0] = x[0::2]
        z[1:, 1] = x[1::2]
        
        d = compute_distances(z)
        if len(d) == 0:
            return 0.0
        
        # Soft min of distances
        # min(d) approx -1/t * log(sum(exp(-t*d)))
        # To avoid overflow/underflow, shift by min(d)
        d_min_val = np.min(d)
        # exp(-t*(d - d_min))
        exp_terms = np.exp(-t * (d - d_min_val))
        soft_min = d_min_val - (1.0/t) * np.log(np.sum(exp_terms))
        
        # Constraint: max distance <= 1
        d_max_val = np.max(d)
        penalty = 100.0 * max(0.0, d_max_val - 1.0)**2
        
        # We want to maximize soft_min, so minimize -soft_min + penalty
        return -soft_min + penalty
    
    # Initial guess: points on a circle?
    # Or random points scaled to unit max distance.
    
    best_score = -np.inf
    best_points = None
    
    # Try a few random restarts
    for trial in range(10):
        np.random.seed(123 + trial)
        # Random points in unit disk
        r = np.sqrt(np.random.rand(15)) * np.random.rand(15)
        theta = 2 * np.pi * np.random.rand(15)
        x0 = np.zeros(30)
        x0[0::2] = r * np.cos(theta)
        x0[1::2] = r * np.sin(theta)
        
        res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
        
        # Reconstruct points
        z = np.zeros((n, 2))
        z[1:, 0] = res.x[0::2]
        z[1:, 1] = res.x[1::2]
        
        d = compute_distances(z)
        d_min_val = np.min(d)
        d_max_val = np.max(d)
        
        if d_max_val > 0:
            ratio = (d_min_val / d_max_val) ** 2
        else:
            ratio = 0.0
        
        if ratio > best_score:
            best_score = ratio
            best_points = z.copy()
            
    # It's possible the optimization didn't fully satisfy d_max=1.
    # We can rescale the best points to have d_max=1 to ensure valid scale, 
    # though the ratio is invariant to scaling.
    if best_points is not None:
        d = compute_distances(best_points)
        d_max_val = np.max(d)
        if d_max_val > 0:
            best_points = best_points / d_max_val
    
    return best_points


# EVOLVE-BLOCK-END
