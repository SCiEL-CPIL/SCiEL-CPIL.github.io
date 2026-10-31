"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16
D = 2

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


def construct():
    """
    Place 16 distinct points in 2D Euclidean space to minimize the squared ratio
    of maximum to minimum pairwise distance.
    
    Approach: Represent 16 points as vertices of a convex polygon defined by a radial
    function r(theta) parameterized by a low-order Fourier series with only even harmonics
    to enforce central symmetry: r(theta) = 1 + sum_{k=1}^{K} a_k * cos(2k*theta)
    
    Use differential evolution for global search, then L-BFGS-B for local refinement.
    """
    
    K = 8  # Number of even harmonics
    
    def points_from_params(params):
        """Convert Fourier parameters to 16 points in 2D."""
        thetas = np.linspace(0, 2 * np.pi, N, endpoint=False)
        radii = np.ones(N)
        for k in range(K):
            radii += params[k] * np.cos(2 * (k + 1) * thetas)
        
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
        dists_sq = dx**2 + dy**2
        
        # Get upper triangle (excluding diagonal)
        iu = np.triu_indices(n, k=1)
        pair_dists_sq = dists_sq[iu]
        
        min_d_sq = np.min(pair_dists_sq)
        max_d_sq = np.max(pair_dists_sq)
        
        if min_d_sq < 1e-12:
            return 1e6  # Penalty for degenerate configurations
        
        ratio_sq = max_d_sq / min_d_sq
        return ratio_sq
    
    # Bounds for each Fourier coefficient
    bounds = [(-0.5, 0.5)] * K
    
    # Global search with differential evolution
    np.random.seed(42)
    de_result = differential_evolution(
        objective, bounds, 
        seed=42, maxiter=200, popsize=30,
        tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
        polish=False
    )
    
    best_x = de_result.x
    best_val = de_result.fun
    
    # Local refinement with L-BFGS-B
    res = minimize(objective, best_x, method='L-BFGS-B',
                   bounds=bounds,
                   options={'maxiter': 10000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    if res.fun < best_val:
        best_val = res.fun
        best_x = res.x
    
    # Additional Nelder-Mead refinement
    res_nm = minimize(objective, best_x, method='Nelder-Mead',
                      options={'maxiter': 20000, 'xatol': 1e-12, 'fatol': 1e-12})
    
    if res_nm.fun < best_val:
        best_val = res_nm.fun
        best_x = res_nm.x
    
    pts = points_from_params(best_x)
    
    # Verify points are distinct
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-8:
                pts[j] += np.array([1e-6, 1e-6])
    
    return pts

# EVOLVE-BLOCK-END
