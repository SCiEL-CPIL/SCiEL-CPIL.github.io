"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import itertools


def construct():
    """
    Place 16 distinct points in 2D Euclidean space to minimize the squared ratio
    of maximum to minimum pairwise distance.
    
    Approach: Represent 16 points as vertices of a convex polygon defined by a radial
    function r(theta) parameterized by a low-order Fourier series with only even harmonics
    to enforce central symmetry. This restricts the search to a low-dimensional manifold
    of star-shaped convex bodies.
    
    We use 16 points evenly spaced in angle, with radii given by:
    r(theta) = 1 + a*cos(2*theta) + b*cos(4*theta) + c*cos(6*theta)
    
    The parameters (a, b, c) are optimized to minimize (max_dist / min_dist)^2.
    """
    
    def points_from_params(params):
        """Convert Fourier parameters to 16 points in 2D."""
        a, b, c = params
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        # Radial function with even harmonics (central symmetry)
        radii = 1.0 + a * np.cos(2 * thetas) + b * np.cos(4 * thetas) + c * np.cos(6 * thetas)
        
        # Ensure radii are positive (star-shaped)
        radii = np.maximum(radii, 0.1)
        
        x = radii * np.cos(thetas)
        y = radii * np.sin(thetas)
        return np.column_stack([x, y])
    
    def objective(params):
        """Compute the squared ratio of max to min pairwise distance."""
        pts = points_from_params(params)
        
        # Compute all pairwise distances
        n = len(pts)
        dx = pts[:, np.newaxis, 0] - pts[np.newaxis, :, 0]
        dy = pts[:, np.newaxis, 1] - pts[np.newaxis, :, 1]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        min_d = np.min(pair_dists)
        max_d = np.max(pair_dists)
        
        if min_d < 1e-10:
            return 1e6  # Penalty for degenerate configurations
        
        ratio = max_d / min_d
        return ratio ** 2
    
    # Initial guess: circle (a=b=c=0)
    x0 = np.array([0.0, 0.0, 0.0])
    
    # Try multiple starting points to avoid local minima
    best_x = x0.copy()
    best_val = objective(x0)
    
    # Random perturbations
    np.random.seed(42)
    for trial in range(20):
        if trial == 0:
            x_try = x0.copy()
        else:
            x_try = np.random.uniform(-0.5, 0.5, size=3)
        
        res = minimize(objective, x_try, method='Nelder-Mead',
                       options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
    
    # Refine with more iterations
    res = minimize(objective, best_x, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
    
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x
    
    pts = points_from_params(best_x)
    
    # Verify points are distinct
    # If any points are too close, add small perturbation
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-8:
                pts[j] += np.array([1e-6, 1e-6])
    
    return pts

# EVOLVE-BLOCK-END
