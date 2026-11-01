"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution


def _objective(points_flat):
    """Objective function: minimize the squared ratio of max to min pairwise distance."""
    points = points_flat.reshape(N, D)
    # Compute all pairwise distances
    # Use broadcasting to compute distance matrix
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    # Get upper triangle indices to avoid zeros on diagonal
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    
    if len(dist_sq_upper) == 0:
        return 1e10
    
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    if min_d_sq <= 1e-12:
        return 1e10
    
    ratio = max_d_sq / min_d_sq
    return ratio


def construct():
    """
    Construct 14 points in 3D space to minimize the squared ratio of
    maximum to minimum pairwise distance.
    
    Uses differential evolution to search for an optimal configuration.
    """
    # Initial guess: use a known good configuration or random
    # Let's try starting from a regular polytope-like structure or just random
    
    # We'll use differential evolution for optimization
    # Bounds: let's center around origin, scale doesn't matter for ratio
    # But we need to avoid zero distances
    
    # Strategy: normalize the points so that the minimum distance is 1,
    # then minimize the maximum distance. Or just minimize the ratio directly.
    
    # Differential evolution approach
    bounds = [(-10, 10)] * (N * D)
    
    # Try multiple restarts with different initial populations
    best_result = None
    best_score = np.inf
    
    # First, try a structured initial guess
    # Use vertices of a regular tetrahedron expanded, or just use random
    
    # Let's use differential evolution with a reasonable number of iterations
    # 120 seconds limit, so we need to be efficient
    
    # Reduce the search space by fixing one point at origin and another on x-axis
    # This reduces symmetry: N*D - 4 degrees of freedom
    # Point 0 at (0,0,0), Point 1 at (1,0,0)
    
    def objective_reduced(flat):
        # flat has (N-2)*D parameters
        points = np.zeros((N, D))
        points[0] = [0, 0, 0]
        points[1] = [1, 0, 0]
        points[2:] = flat.reshape(N-2, D)
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(N, k=1)
        dist_sq_upper = dist_sq[iu]
        
        min_d_sq = np.min(dist_sq_upper)
        max_d_sq = np.max(dist_sq_upper)
        
        if min_d_sq <= 1e-12:
            return 1e10
        
        return max_d_sq / min_d_sq
    
    bounds_reduced = [(-5, 5)] * ((N-2) * D)
    
    # Run differential evolution
    # maxiter should be reasonable to fit in time
    result = differential_evolution(
        objective_reduced,
        bounds_reduced,
        maxiter=200,
        popsize=15,
        tol=1e-6,
        seed=42,
        polish=True
    )
    
    if result.success or result.fun < best_score:
        best_score = result.fun
        best_result = result.x
    
    # Convert back to full points
    points = np.zeros((N, D))
    points[0] = [0, 0, 0]
    points[1] = [1, 0, 0]
    points[2:] = best_result.reshape(N-2, D)
    
    # Verify and return
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    iu = np.triu_indices(N, k=1)
    dist_sq_upper = dist_sq[iu]
    min_d_sq = np.min(dist_sq_upper)
    max_d_sq = np.max(dist_sq_upper)
    
    # The score is (max/min)^2 = max_sq / min_sq
    score = max_d_sq / min_d_sq
    
    return points
# EVOLVE-BLOCK-END
