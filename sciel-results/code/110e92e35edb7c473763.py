# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv, gamma
import time

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes and weights in x = cos(theta) in [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl is in [-1,1], w_gl are weights (sum = 2)
    # theta = arccos(x), dOmega = sin(theta) dtheta dphi = dx dphi
    # So integral over sphere = integral_{-1}^{1} dx integral_0^{2pi} dphi
    # with weight w_gl for x and 2pi/130 for phi
    n_phi = 130
    phi = 2.0 * np.pi * np.arange(n_phi) / n_phi
    dphi = 2.0 * np.pi / n_phi
    return x_gl, w_gl, phi, dphi

def _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=10):
    """
    Build the matrix Y[i,j] where i indexes quadrature points (n_gl * n_phi)
    and j indexes m = -k..k.
    Y_0 = N_0 * P_k^0(cos theta)
    Y_m = sqrt(2) * N_m * P_k^m(cos theta) * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_k^|m|(cos theta) * sin(|m|*phi) for m < 0
    N_m = sqrt( (2k+1)/(4*pi) * (k-m)!/(k+m)! )
    """
    n_gl = len(x_gl)
    n_phi = len(phi)
    n_m = 2 * k + 1
    Y = np.zeros((n_gl * n_phi, n_m), dtype=np.float64)
    
    for m_idx in range(n_m):
        m = m_idx - k
        abs_m = abs(m)
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4.0 * np.pi) * 
                       float(gamma(k - abs_m + 1)) / float(gamma(k + abs_m + 1)))
        # Associated Legendre with Condon-Shortley phase
        # lpmv(m, k, x) includes the Condon-Shortley phase (-1)^m
        P = lpmv(abs_m, k, x_gl)  # shape (n_gl,)
        
        if m == 0:
            vals = N_m * P  # shape (n_gl,)
            # Tile for phi
            Y[:, m_idx] = np.repeat(vals, n_phi)
        elif m > 0:
            cos_m = np.cos(m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * cos_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
        else:
            sin_m = np.sin(abs_m * phi)  # shape (n_phi,)
            vals = np.sqrt(2.0) * N_m * P[:, np.newaxis] * sin_m[np.newaxis, :]
            Y[:, m_idx] = vals.ravel()
    
    return Y

def _compute_ratio(coeffs, Y, w_gl, dphi):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    f = Y @ coeffs  # shape (n_points,)
    # Weights: each point has weight w_gl[i] * dphi
    # For point index p = i * n_phi + j, weight = w_gl[i] * dphi
    n_gl = len(w_gl)
    n_phi = Y.shape[0] // n_gl
    weights = np.repeat(w_gl, n_phi) * dphi
    # ||f||_2^2 = sum w * f^2
    # ||f||_4^2 = sum w * f^4
    f2 = f * f
    f4 = f2 * f2
    norm2_sq = np.dot(weights, f2)
    norm4_sq = np.dot(weights, f4)
    if norm2_sq <= 0:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(norm4_sq)
    return norm4 / norm2

def _nelder_mead_minimize(func, x0, bounds=None, max_iter=5000, tol=1e-8):
    """Simple Nelder-Mead implementation."""
    n = len(x0)
    alpha = 1.0
    gamma = 2.0
    rho = 0.5
    sigma = 0.5
    
    # Initialize simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0
    for i in range(n):
        simplex[i + 1] = x0.copy()
        simplex[i + 1][i] += 0.1 * abs(x0[i]) + 0.01
    
    fvals = np.array([func(simplex[i]) for i in range(n + 1)])
    
    best_val = fvals[0]
    best_x = x0.copy()
    
    for it in range(max_iter):
        # Sort simplex by function value
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        
        if fvals[0] < best_val:
            best_val = fvals[0]
            best_x = simplex[0].copy()
        
        # Check convergence
        if np.max(fvals) - np.min(fvals) < tol * max(1.0, abs(fvals[0])):
            break
        
        # Centroid of all but worst
        centroid = np.mean(simplex[:-1], axis=0)
        
        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        
        if fr < fvals[0]:
            # Expansion
            xe = centroid + gamma * (centroid - simplex[-1])
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                fvals[-1] = fe
            else:
                simplex[-1] = xr
                fvals[-1] = fr
        elif fr < fvals[-2]:
            simplex[-1] = xr
            fvals[-1] = fr
        else:
            # Contraction
            if fr < fvals[-1]:
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc <= fr:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
            else:
                xc = centroid - rho * (centroid - simplex[-1])
                fc = func(xc)
                if fc <= fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
    
    return best_x, best_val

def rudin_problem10() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 620.0  # safety margin
    
    k = 10
    x_gl, w_gl, phi, dphi = _build_quadrature()
    Y = _spherical_harmonics_matrix(x_gl, w_gl, phi, dphi, k=k)
    
    # Objective function
    def objective(coeffs):
        return _compute_ratio(coeffs, Y, w_gl, dphi)
    
    # Start with all ones
    x0 = np.ones(2 * k + 1, dtype=np.float64)
    f0 = objective(x0)
    
    best_coeffs = x0.copy()
    best_val = f0
    
    # Try a few different initializations and optimize
    n_restarts = 5
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            # Random perturbation
            x_init = x0 + 0.5 * np.random.randn(2 * k + 1)
        
        x_opt, f_opt = _nelder_mead_minimize(
            objective, x_init, max_iter=3000, tol=1e-9
        )
        
        if f_opt < best_val:
            best_val = f_opt
            best_coeffs = x_opt.copy()
        
        if time.time() > deadline - 60.0:
            break
    
    # Validate
    if len(best_coeffs) != 21:
        return np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return np.ones(21)
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return np.ones(21)
    
    return best_coeffs

# EVOLVE-BLOCK-END
