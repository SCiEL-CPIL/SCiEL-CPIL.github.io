# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, factorial
import time


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    # Gauss-Legendre nodes and weights for 70 points on [-1, 1]
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    # x_gl = cos(theta), w_gl are weights for integral over [-1,1]
    # Surface measure: dOmega = dphi * d(cos theta), so weights are w_gl * dphi
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    # Full weights: w_gl[i] * dphi for each (i, j)
    # Shape: (70, 130)
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl  # (70,)
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    Y_0 = N_0 * P_k^0
    Y_m = sqrt(2) * N_m * P_k^m * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_k^|m| * sin(|m|*phi) for m < 0
    N_m = sqrt((2k+1)/(4*pi) * (k-m)!/(k+m)!)
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        # Normalization
        N_m = np.sqrt((2 * k + 1) / (4 * np.pi) * 
                      float(factorial(k - m_abs) / factorial(k + m_abs)))
        
        # Associated Legendre polynomial with Condon-Shortley phase
        # lpmv(m, l, x) includes the Condon-Shortley phase (-1)^m
        P_m = lpmv(m_abs, k, cos_theta)  # shape (n_theta,)
        
        if m_abs == 0:
            Y[0] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            # m = +m_abs
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            # m = -m_abs
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics and weights."""
    # f = sum_m c_m * Y_m, shape (n_theta, n_phi)
    f = np.tensordot(coeffs, Y, axes=([0], [0]))  # (n_theta, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))  # ||f||_4 = (integral |f|^4)^{1/4}
    
    return norm4 / norm2


def _objective(coeffs, Y, weights):
    """Objective function for optimization."""
    return _compute_ratio(coeffs, Y, weights)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 30.0  # seconds
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Since R is homogeneous, fix ||c||_2 = 1
    # Start with multiple initial guesses
    
    best_coeffs = np.ones(21, dtype=np.float64)
    best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Try several initial conditions
    initial_guesses = []
    
    # 1. All ones (normalized)
    c0 = np.ones(21) / np.sqrt(21)
    initial_guesses.append(c0)
    
    # 2. Single highest |m| mode (m=10 cos)
    c1 = np.zeros(21)
    c1[20] = 1.0  # m=10
    initial_guesses.append(c1)
    
    # 3. Single m=10 sin
    c2 = np.zeros(21)
    c2[10] = 1.0  # m=-10
    initial_guesses.append(c2)
    
    # 4. Symmetric cos modes: m=0, 2, 4, 6, 8, 10
    c3 = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c3[10 + m] = 1.0
    c3 /= np.linalg.norm(c3)
    initial_guesses.append(c3)
    
    # 5. Random starts
    rng = np.random.default_rng(42)
    for i in range(5):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    # Evaluate initial guesses
    for c_init in initial_guesses:
        r = _compute_ratio(c_init, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Use L-BFGS-B optimization from the best initial guess
    # We optimize on the sphere ||c||=1 by using a parameterization or constraint
    # Simpler: optimize without constraint, then normalize at the end
    # But the ratio is homogeneous, so we can add a penalty for ||c|| != 1
    
    def objective_constrained(c):
        if time.time() > deadline - safety_margin:
            return _compute_ratio(best_coeffs, Y, weights)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10
        c_normalized = c / c_norm
        return _compute_ratio(c_normalized, Y, weights)
    
    # Run optimization from the best initial guess
    result = minimize(
        objective_constrained,
        best_coeffs,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    
    if result.success or (result.fun < best_ratio):
        c_opt = result.x / np.linalg.norm(result.x)
        r_opt = _compute_ratio(c_opt, Y, weights)
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt
    
    # Try a few more random restarts if time permits
    for i in range(10):
        if time.time() > deadline - safety_margin:
            break
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        result = minimize(
            objective_constrained,
            c_rand,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale to be in valid range
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
