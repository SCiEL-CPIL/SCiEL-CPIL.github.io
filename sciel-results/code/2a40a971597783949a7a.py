# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a constrained optimization approach where points are constrained to the unit sphere
    and the objective is to minimize the variance of pairwise distances. This encourages
    all pairwise distances to be as equal as possible, which maximizes the d_min/d_max ratio.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3
    
    # Precompute indices for upper triangle of distance matrix
    iu = np.triu_indices(n, k=1)
    n_pairs = len(iu[0])
    
    def pairwise_distances(pts):
        """Compute all pairwise distances from a flat array."""
        pts = pts.reshape(n, d)
        diff = pts[iu[0]] - pts[iu[1]]
        dists = np.sqrt(np.sum(diff**2, axis=1))
        return dists

    def objective(x):
        """
        Minimize the variance of pairwise distances.
        x is a flat array of shape (n*d,).
        """
        pts = x.reshape(n, d)
        dists = pairwise_distances(x)
        mean_d = np.mean(dists)
        if mean_d < 1e-10:
            return 1e10
        # Variance is scale invariant if we normalize, but we are on unit sphere so scale is fixed.
        # Minimizing variance minimizes the spread.
        return np.sum((dists - mean_d)**2)

    def constraints():
        """
        Constraints: each point must lie on the unit sphere.
        """
        cons = []
        for i in range(n):
            # Define constraint function for point i
            def con_func(x, idx=i):
                pts = x.reshape(n, d)
                return np.sum(pts[idx]**2) - 1.0
            cons.append({'type': 'eq', 'fun': con_func})
        return cons

    best_pts = None
    best_score = -np.inf
    
    # Run multiple trials with different random initializations
    np.random.seed(42)
    
    for trial in range(20):
        # Random points on the unit sphere
        x0 = np.random.randn(n, d)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        # Use SLSQP to handle the equality constraints
        # maxiter increased to ensure convergence
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            constraints=constraints(),
            options={'maxiter': 1000, 'ftol': 1e-10}
        )
        
        if result.success or result.fun < 1e-5:
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10 and np.max(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # Fallback if optimization failed or didn't improve
    if best_pts is None:
        # Use a known good configuration or random
        # For 14 points, a random distribution on sphere is a decent baseline
        np.random.seed(123)
        best_pts = np.random.randn(n, d)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    else:
        # Ensure the returned points are normalized to unit sphere for consistency, 
        # though the constraint should have kept them there.
        # Re-normalizing might slightly break the distance ratio if the optimizer 
        # found a non-unit sphere solution due to tolerance, but SLSQP eq constraints are strict.
        pass

    return best_pts


# EVOLVE-BLOCK-END
