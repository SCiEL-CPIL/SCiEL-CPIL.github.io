# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the points.

    """
    n = 16
    d = 2

    # Temperature parameter for soft-min/soft-max
    # Smaller T gives sharper approximation to true min/max
    T = 0.05

    def soft_min(x, T):
        """Smooth approximation of min(x)"""
        return -T * np.log(np.sum(np.exp(-x / T)))

    def soft_max(x, T):
        """Smooth approximation of max(x)"""
        return T * np.log(np.sum(np.exp(x / T)))

    def objective_and_grad(x_flat):
        """
        Objective: maximize the soft-min distance after normalizing so that soft-max distance = 1.
        Returns (value, gradient) where value is negative of what we want to maximize.
        """
        x = x_flat.reshape(n, d)
        
        # Compute pairwise distances
        dists = pdist(x)
        
        # Handle degenerate case
        if np.allclose(dists, 0):
            return 0.0, np.zeros_like(x_flat)
        
        # Soft-max distance (approximation of max distance)
        D = soft_max(dists, T)
        
        # If D is too small, points are nearly coincident
        if D < 1e-10:
            return 0.0, np.zeros_like(x_flat)
        
        # Normalize: scale points so that soft-max distance = 1
        # y = x / D
        y = x / D
        
        # Compute distances of normalized points
        dists_y = pdist(y)
        
        # Soft-min distance of normalized points
        m = soft_min(dists_y, T)
        
        # We want to maximize m, so return -m
        value = -m
        
        # Compute gradient numerically (since analytic gradient is complex)
        # Use finite differences for the gradient
        grad = np.zeros_like(x_flat)
        eps = 1e-7
        for i in range(len(x_flat)):
            x_plus = x_flat.copy()
            x_minus = x_flat.copy()
            x_plus[i] += eps
            x_minus[i] -= eps
            grad[i] = (objective_and_grad_value(x_plus) - objective_and_grad_value(x_minus)) / (2 * eps)
        
        return value, grad
    
    def objective_and_grad_value(x_flat):
        """Helper to compute just the objective value (for gradient computation)"""
        x = x_flat.reshape(n, d)
        dists = pdist(x)
        if np.allclose(dists, 0):
            return 0.0
        D = soft_max(dists, T)
        if D < 1e-10:
            return 0.0
        y = x / D
        dists_y = pdist(y)
        m = soft_min(dists_y, T)
        return -m

    def objective_only(x_flat):
        """Just the objective value for L-BFGS-B"""
        val, _ = objective_and_grad(x_flat)
        return val

    def objective_grad_only(x_flat):
        """Just the gradient for L-BFGS-B"""
        _, grad = objective_and_grad(x_flat)
        return grad

    best_points = None
    best_score = -np.inf

    # Multi-start with different initializations
    for seed in range(30):
        rng = np.random.RandomState(seed)
        
        # Different initialization strategies
        if seed % 3 == 0:
            # Random points in unit circle
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.5, 1.0, n)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        elif seed % 3 == 1:
            # Points on a circle with some perturbation
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.uniform(-0.1, 0.1, n)
            radii = np.ones(n) + rng.uniform(-0.1, 0.1, n)
            x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Random points, then normalize
            x0 = rng.randn(n, d)
            dists = pdist(x0)
            D = np.max(dists)
            if D > 0:
                x0 = x0 / D
        
        x0_flat = x0.flatten()
        
        # Run L-BFGS-B
        try:
            res = minimize(
                objective_only,
                x0_flat,
                jac=objective_grad_only,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            points = res.x.reshape(n, d)
            
            # Evaluate the actual score
            dists = pdist(points)
            if len(dists) > 0 and np.max(dists) > 0:
                m = np.min(dists)
                M = np.max(dists)
                score = (m / M) ** 2
            else:
                score = 0
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            continue

    # If all optimizations failed, use a reasonable fallback
    if best_points is None:
        # Use points on a regular polygon (hexadecagon) as fallback
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])

    return best_points


# EVOLVE-BLOCK-END
