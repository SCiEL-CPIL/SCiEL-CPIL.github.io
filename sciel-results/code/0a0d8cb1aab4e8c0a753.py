"""Spectral energy minimization for covering the sphere with 130 points."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 130

# EVOLVE-BLOCK-START
def _pair_correlation_energy(points, max_l=20):
    """
    Compute a spectral energy based on the variance of the pair correlation function.
    The power at multipole l is proportional to sum_{i,j} P_l(x_i . x_j).
    For a uniform distribution, this sum is approximately N (from i=j terms).
    We minimize the squared deviation from N for l >= 1.
    """
    dots = points @ points.T
    np.clip(dots, -1.0, 1.0, out=dots)
    
    energy = 0.0
    for l in range(1, max_l + 1):
        # Compute P_l(cos(theta_ij)) for all pairs
        # legvander returns a matrix where row i is [P_0(d_i), P_1(d_i), ..., P_l(d_i)]
        P_l = np.polynomial.legendre.legvander(dots, l)[:, -1]
        # Sum over all pairs (i, j)
        # The diagonal terms are P_l(1) = 1, there are N of them
        pair_sum = np.sum(P_l)
        # Deviation from the uniform case (where off-diagonal terms average to 0)
        # Uniform sum is N.
        deviation = pair_sum - N
        # Weight by 1/l to prioritize low-order modes (large-scale uniformity)
        weight = 1.0 / l
        energy += weight * deviation**2
    
    return energy


def _refine_from_spectral(start_points, max_iters=1000, tol=1e-12):
    """
    Refine points using spectral energy minimization with Powell's method.
    """
    # Convert to (theta, phi)
    theta = np.arccos(np.clip(start_points[:, 2], -1.0, 1.0))
    phi = np.arctan2(start_points[:, 1], start_points[:, 0])
    
    # Flatten to 1D array of 2N parameters
    x0 = np.concatenate([theta, phi])
    
    def obj(x):
        t = x[:N]
        p = x[N:]
        # Wrap phi to [0, 2pi)
        p = np.mod(p, 2*np.pi)
        # Convert back to Cartesian
        pts = np.column_stack([
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])
        return _pair_correlation_energy(pts, max_l=12)
    
    # Use Powell for local refinement, which is more robust than Nelder-Mead for higher dimensions
    result = minimize(obj, x0, method='Powell', 
                      options={'maxiter': max_iters, 'xtol': tol, 'ftol': tol})
    
    t = result.x[:N]
    p = np.mod(result.x[N:], 2*np.pi)
    pts = np.column_stack([
        np.sin(t) * np.cos(p),
        np.sin(t) * np.sin(p),
        np.cos(t)
    ])
    # Normalize
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts, result.fun


def construct():
    # Generate multiple initial configurations to avoid local minima
    best_pts = None
    best_energy = float('inf')
    
    # Try a few different Fibonacci spiral offsets
    for offset in [0.0, 0.1, 0.2, 0.5]:
        i = np.arange(N) + 0.5 + offset
        z = 1.0 - 2.0 * i / N
        # Clip z to ensure it's within [-1, 1]
        z = np.clip(z, -1.0, 1.0)
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        start_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Refine using spectral energy minimization
        refined_pts, energy = _refine_from_spectral(start_pts, max_iters=500)
        
        # Normalize to ensure unit sphere
        norms = np.linalg.norm(refined_pts, axis=1, keepdims=True)
        refined_pts = refined_pts / norms
        
        if energy < best_energy:
            best_energy = energy
            best_pts = refined_pts
    
    return best_pts
# EVOLVE-BLOCK-END
