# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    
    Uses a constrained optimization approach to make all pairwise distances as equal as possible,
    which maximizes the d_min/d_max ratio.
    
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
        """Minimize variance of pairwise distances (scale-invariant)."""
        dists = pairwise_distances(x)
        if np.any(dists < 1e-10):
            return 1e10
        mean_d = np.mean(dists)
        if mean_d < 1e-10:
            return 1e10
        # Coefficient of variation squared - scale invariant
        return np.var(dists) / (mean_d ** 2)
    
    def objective_grad(x):
        """Gradient of the variance objective."""
        pts = x.reshape(n, d)
        dists = pairwise_distances(x)
        mean_d = np.mean(dists)
        
        if mean_d < 1e-10:
            return np.zeros_like(x)
        
        # Gradient of variance w.r.t. each coordinate
        # var = (1/m) * sum((d_i - mean)^2)
        # d(var)/d(d_i) = (2/m) * (d_i - mean)
        # d(d_i)/d(x_j) = (x_j_i - x_j_k) / d_i  for the pair (i,k)
        
        grad = np.zeros_like(x)
        m = n_pairs
        for idx in range(m):
            i, j = iu[0][idx], iu[1][idx]
            d_i = dists[idx]
            if d_i < 1e-10:
                continue
            # d(d_i)/d(pts[i]) = (pts[i] - pts[j]) / d_i
            # d(d_i)/d(pts[j]) = (pts[j] - pts[i]) / d_i
            diff = pts[i] - pts[j]
            factor = (2.0 / m) * (d_i - mean_d) / d_i
            grad[i * d:(i + 1) * d] += factor * diff
            grad[j * d:(j + 1) * d] -= factor * diff
        
        # Additional term from derivative of mean_d
        # mean_d = (1/m) * sum(d_i)
        # d(mean_d)/d(x) = (1/m) * sum of d(d_i)/d(x)
        # The full gradient of var/mean^2 is more complex, but for BFGS we can use numerical or approximate
        # For simplicity, let's use BFGS which doesn't require exact gradient
        
        return grad
    
    # Initial guess: use a good starting configuration
    # Try multiple initializations and pick the best
    
    best_pts = None
    best_score = -np.inf
    
    # Initialization 1: Random on sphere
    np.random.seed(42)
    for trial in range(5):
        x0 = np.random.randn(n, d)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        x0 = x0 / norms
        x0_flat = x0.flatten()
        
        # Use SLSQP for better convergence
        result = minimize(
            objective, 
            x0_flat, 
            method='SLSQP',
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if result.success:
            pts = result.x.reshape(n, d)
            if np.all(np.isfinite(pts)):
                dists = pairwise_distances(result.x)
                if np.min(dists) > 1e-10:
                    score = (np.min(dists) / np.max(dists)) ** 2
                    if score > best_score:
                        best_score = score
                        best_pts = pts
    
    # Initialization 2: Fibonacci sphere
    golden_ratio = (1 + np.sqrt(5)) / 2
    theta = 2 * np.pi * np.arange(n) / golden_ratio
    phi = np.arccos(1 - 2 * (np.arange(n) + 0.5) / n)
    
    x0 = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    x0_flat = x0.flatten()
    
    result = minimize(
        objective,
        x0_flat,
        method='SLSQP',
        options={'maxiter': 500, 'ftol': 1e-12}
    )
    
    if result.success:
        pts = result.x.reshape(n, d)
        if np.all(np.isfinite(pts)):
            dists = pairwise_distances(result.x)
            if np.min(dists) > 1e-10:
                score = (np.min(dists) / np.max(dists)) ** 2
                if score > best_score:
                    best_score = score
                    best_pts = pts
    
    # If no good result, fallback
    if best_pts is None:
        np.random.seed(123)
        best_pts = np.random.randn(n, d)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
    
    return best_pts


# EVOLVE-BLOCK-END
