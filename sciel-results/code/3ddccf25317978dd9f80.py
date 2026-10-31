"""Fourier-invariant Hermite-Gaussian test function: minimize the C4 uncertainty
upper bound via direct angular-sphere minimization with local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _c4_value(a0, a4, a8):
    """Compute the C4 upper bound for coefficients (a0, a4, a8).
    
    g(x) = (a0*H0 + a4*H4 + a8*H8 + a12*H12) * exp(-x^2/2)
    with a12 = -(a0 + a4 + a8) so g(0) = 0.
    
    The C4 bound is (1/8)*(1 - 4*<x^2>)*(1 - 2*<xi^2>)^2 / ||g||^2
    where expectations are with respect to |g|^2 / ||g||^2.
    """
    a = np.array([a0, a4, a8])
    a12 = -(a0 + a4 + a8)
    c = np.array([a0, a4, a8, a12])
    # Degrees: 0, 4, 8, 12
    k = np.array([0, 4, 8, 12], dtype=float)
    
    norm2 = np.sum(c**2)
    if norm2 < 1e-30:
        return 1e10  # trivial
    
    # <x^2> = sum c_k^2 * (k+1) / norm2
    x2 = np.sum(c**2 * (k + 1.0)) / norm2
    # <xi^2> = sum c_k^2 * (k + 0.5) / norm2
    xi2 = np.sum(c**2 * (k + 0.5)) / norm2
    
    # C4 = (1/8) * (1 - 4*x2) * (1 - 2*xi2)^2
    val = 0.125 * (1.0 - 4.0 * x2) * (1.0 - 2.0 * xi2)**2
    return val


def _c4_vec(a):
    """Vectorized C4 for a 1D array of 3 coefficients."""
    return _c4_value(a[0], a[1], a[2])


def construct():
    t_start = time.time()
    DEADLINE = 200.0  # safety margin under the 210 s kill
    
    try:
        # The problem is homogeneous: C4(λa) = C4(a) for λ > 0.
        # So we minimize over directions on the unit sphere S^2.
        # Parameterize: a0 = sin(θ)cos(φ), a4 = sin(θ)sin(φ), a8 = cos(θ)
        # θ ∈ [0, π], φ ∈ [0, 2π)
        
        best_val = _c4_value(-113.0/100.0, 1.0/25.0, 1.0/3240.0)
        best_a = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
        
        # Dense angular grid for global search
        n_theta = 80
        n_phi = 160
        thetas = np.linspace(0.0, np.pi, n_theta)
        phis = np.linspace(0.0, 2.0*np.pi, n_phi, endpoint=False)
        
        # Vectorized evaluation over the grid
        T, P = np.meshgrid(thetas, phis, indexing='ij')
        a0_grid = np.sin(T) * np.cos(P)
        a4_grid = np.sin(T) * np.sin(P)
        a8_grid = np.cos(T)
        
        # Compute C4 on the grid
        a12_grid = -(a0_grid + a4_grid + a8_grid)
        c0 = a0_grid
        c4 = a4_grid
        c8 = a8_grid
        c12 = a12_grid
        
        norm2_grid = c0**2 + c4**2 + c8**2 + c12**2
        # Avoid division by zero
        norm2_grid = np.where(norm2_grid < 1e-30, 1e-30, norm2_grid)
        
        x2_grid = (c0**2 * 1.0 + c4**2 * 5.0 + c8**2 * 9.0 + c12**2 * 13.0) / norm2_grid
        xi2_grid = (c0**2 * 0.5 + c4**2 * 4.5 + c8**2 * 8.5 + c12**2 * 12.5) / norm2_grid
        
        c4_grid = 0.125 * (1.0 - 4.0 * x2_grid) * (1.0 - 2.0 * xi2_grid)**2
        
        # Find the minimum on the grid
        min_idx = np.unravel_index(np.argmin(c4_grid), c4_grid.shape)
        min_theta = thetas[min_idx[0]]
        min_phi = phis[min_idx[1]]
        
        # Local polish from the best grid point and several other starting points
        start_points = []
        # Best grid point
        start_points.append([np.sin(min_theta)*np.cos(min_phi),
                            np.sin(min_theta)*np.sin(min_phi),
                            np.cos(min_theta)])
        # Add a few diverse starting points
        for trial_theta in [0.3, 0.7, 1.2, 1.8, 2.5, 3.0]:
            for trial_phi in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
                start_points.append([np.sin(trial_theta)*np.cos(trial_phi),
                                    np.sin(trial_theta)*np.sin(trial_phi),
                                    np.cos(trial_phi)])
        # Also add the known-good seed point normalized
        seed_a = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
        seed_norm = np.linalg.norm(seed_a)
        if seed_norm > 1e-15:
            start_points.append(seed_a / seed_norm)
        
        # Local minimization from each starting point
        for sp in start_points:
            if time.time() - t_start > DEADLINE:
                break
            try:
                res = minimize(_c4_vec, sp, method='Nelder-Mead',
                              options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.success and res.fun < best_val:
                    best_val = res.fun
                    best_a = res.x.copy()
            except Exception:
                pass
        
        # Final high-precision polish with BFGS if we found an improvement
        if best_val < _c4_value(-113.0/100.0, 1.0/25.0, 1.0/3240.0) - 1e-15:
            if time.time() - t_start < DEADLINE:
                try:
                    res2 = minimize(_c4_vec, best_a, method='BFGS',
                                   options={'maxiter': 10000, 'gtol': 1e-14})
                    if res2.fun < best_val:
                        best_val = res2.fun
                        best_a = res2.x.copy()
                except Exception:
                    pass
        
        return best_a
    except Exception:
        # Fallback to the known-good prior
        return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
# EVOLVE-BLOCK-END
