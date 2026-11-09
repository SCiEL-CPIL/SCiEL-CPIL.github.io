# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    # Objective: minimize variance of pairwise distances (make them as equal as possible)
    # This corresponds to finding a spherical code with constant chord length.
    # We use a penalty/optimization approach to minimize (max_dist - min_dist) or similar.
    # A smoother approach: minimize the sum of squared deviations from the mean distance,
    # or directly minimize the range. However, range is non-smooth.
    # Let's try minimizing the variance of pairwise distances.
    
    def pairwise_distances(pts):
        # pts is (n, d)
        # Compute all pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        # Extract upper triangle
        iu = np.triu_indices(n, k=1)
        return dists[iu]
    
    def objective(x):
        pts = x.reshape(n, d)
        dists = pairwise_distances(pts)
        # We want to maximize min/max ratio.
        # Equivalent to minimizing max/min.
        # Or minimizing the spread.
        # Let's minimize the variance of the distances.
        # Actually, a better objective for "equal distances" is to minimize
        # sum((d_i - mean_d)^2) which is the variance.
        mean_d = np.mean(dists)
        return np.sum((dists - mean_d)**2)
    
    # Initial guess: points on a sphere, randomly distributed
    # Or use a known configuration like vertices of a polyhedron if possible,
    # but 14 doesn't map to a regular polyhedron easily.
    # Let's start with points on a unit sphere, randomly.
    np.random.seed(123)
    x0 = np.random.randn(n, d)
    # Normalize to unit sphere
    norms = np.linalg.norm(x0, axis=1, keepdims=True)
    x0 = x0 / norms
    x0_flat = x0.flatten()
    
    # Use SLSQP or BFGS? SLSQP is good for constrained, but we have no explicit constraints here
    # other than the implicit geometry. The objective is smooth.
    # Let's try BFGS first, it's faster for unconstrained.
    # We need to be careful about local minima.
    # To avoid all points collapsing to one point (which gives 0 distance variance),
    # we should ensure the points are spread out.
    # The variance objective alone might allow all points to be close together.
    # We need to constrain the scale or the mean distance.
    # Let's add a constraint or modify the objective.
    # If all points are close, variance is small.
    # We want points to be far apart AND close together (relative to each other).
    # The ratio d_min/d_max is scale-invariant.
    # So we can fix the scale, e.g., fix the mean distance to 1, or fix the radius of the sphere.
    # Let's fix the points to lie on a sphere of radius R, and optimize the positions on the sphere.
    # Or, simpler: minimize the variance of distances, subject to the constraint that the mean distance is 1.
    # Or, even simpler: minimize the variance of distances, and add a penalty if the mean distance is too small.
    
    # Let's try a different objective: minimize (max_d - min_d) / mean_d
    # This is the relative spread.
    # But max and min are non-smooth.
    
    # Let's stick to variance but add a constraint on the mean distance.
    # Or, use the fact that for a spherical code, the points should be on a sphere.
    # Let's parameterize the points by their spherical coordinates? No, that's messy.
    
    # Let's try this: minimize the variance of the pairwise distances,
    # but initialize with points on a sphere and maybe use a penalty to keep them on a sphere?
    # Actually, if we minimize variance, the points will tend to cluster.
    # We need to encourage them to be spread out.
    
    # Alternative: Maximize the minimum distance, subject to the maximum distance being fixed.
    # This is the classic packing problem.
    # Let's fix the maximum distance to 1 (or some value) and maximize the minimum distance.
    # But max distance is non-smooth.
    
    # Let's go back to the variance idea, but penalize small scales.
    # Objective: variance(dists) + lambda * (1/mean_d)^2
    # This encourages large mean distance and small variance.
    
    def objective2(x):
        pts = x.reshape(n, d)
        dists = pairwise_distances(pts)
        mean_d = np.mean(dists)
        if mean_d < 1e-6:
            return 1e10
        var_d = np.var(dists)
        return var_d / (mean_d**2)  # Coefficient of variation squared
    
    # This objective is scale-invariant and encourages equal distances.
    
    # Let's use BFGS on this objective.
    result = minimize(objective2, x0_flat, method='BFGS', options={'maxiter': 1000})
    
    pts = result.x.reshape(n, d)
    
    # Refine with a second pass, maybe using L-BFGS-B or more iterations
    # Let's try to improve by using the result as initial guess for a more robust method
    # or just check if it's valid.
    
    # Ensure no NaNs or infs
    if not np.all(np.isfinite(pts)):
        # Fallback to random
        pts = np.random.randn(n, d)
        
    return pts


# EVOLVE-BLOCK-END
