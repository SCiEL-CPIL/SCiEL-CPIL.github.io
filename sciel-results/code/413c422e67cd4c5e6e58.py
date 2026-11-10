# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize


def _build_quadrature():
    """Build the 70-point Gauss-Legendre (cos theta) x 130-point phi grid."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    weights = np.outer(w_gl, np.full(n_phi, dphi))
    cos_theta = x_gl
    return cos_theta, phi, weights


def _spherical_harmonics_real(cos_theta, phi, k=10):
    """
    Compute real orthonormal spherical harmonics of degree k.
    Returns array of shape (2*k+1, n_theta, n_phi) where index i corresponds to m = i - k.
    """
    n_theta = len(cos_theta)
    n_phi = len(phi)
    Y = np.zeros((2 * k + 1, n_theta, n_phi))
    
    for m_abs in range(k + 1):
        log_N_sq = np.log((2 * k + 1) / (4 * np.pi)) + gammaln(k + 1 - m_abs) - gammaln(k + 1 + m_abs)
        N_m = np.sqrt(np.exp(log_N_sq))
        
        P_m = lpmv(m_abs, k, cos_theta)
        
        if m_abs == 0:
            Y[k] = N_m * P_m[:, np.newaxis] * np.ones((1, n_phi))
        else:
            Y[k + m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.cos(m_abs * phi)[np.newaxis, :]
            Y[k - m_abs] = np.sqrt(2) * N_m * P_m[:, np.newaxis] * np.sin(m_abs * phi)[np.newaxis, :]
    
    return Y


def _compute_ratio(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 using precomputed harmonics and weights."""
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    return norm4 / norm2


def _compute_ratio_and_grad(coeffs, Y, weights):
    """Compute R and its gradient with respect to coeffs."""
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    
    f2 = f * f
    f4 = f2 * f2
    
    norm2_sq = np.sum(weights * f2)
    norm4_sq = np.sum(weights * f4)
    
    if norm2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_sq))
    
    R = norm4 / norm2
    
    integral_f3_Y = np.tensordot(weights * f**3, Y, axes=([0, 1], [1, 2]))
    integral_f_Y = np.tensordot(weights * f, Y, axes=([0, 1], [1, 2]))
    
    grad = integral_f3_Y / (norm4**3 * norm2) - (norm4 / norm2**3) * integral_f_Y
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 30.0
    
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
    c1[20] = 1.0
    initial_guesses.append(c1)
    
    # 3. Single m=10 sin
    c2 = np.zeros(21)
    c2[10] = 1.0
    initial_guesses.append(c2)
    
    # 4. Symmetric cos modes: m=0, 2, 4, 6, 8, 10
    c3 = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c3[10 + m] = 1.0
    c3 /= np.linalg.norm(c3)
    initial_guesses.append(c3)
    
    # 5. Symmetric sin modes: m=-10, -8, -6, -4, -2
    c4 = np.zeros(21)
    for m in [-10, -8, -6, -4, -2]:
        c4[10 + m] = 1.0
    c4 /= np.linalg.norm(c4)
    initial_guesses.append(c4)
    
    # 6. Only m=0
    c5 = np.zeros(21)
    c5[10] = 1.0
    initial_guesses.append(c5)
    
    # 7. Random starts
    rng = np.random.default_rng(42)
    for i in range(15):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    # Evaluate initial guesses
    for c_init in initial_guesses:
        r = _compute_ratio(c_init, Y, weights)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Use L-BFGS-B optimization with gradient from the best initial guess
    # We optimize on the sphere ||c||=1 by normalizing in the objective
    
    def objective_and_grad(c):
        if time.time() > deadline - safety_margin:
            R, g = _compute_ratio_and_grad(best_coeffs, Y, weights)
            return R, g
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        # Chain rule for normalization
        dot = np.dot(grad, c_normalized)
        grad_c = (grad - c_normalized * dot) / c_norm
        return R, grad_c
    
    # Run optimization from the best initial guess
    result = minimize(
        objective_and_grad,
        best_coeffs,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
    )
    
    if result.success or (result.fun < best_ratio):
        c_opt = result.x / np.linalg.norm(result.x)
        r_opt = _compute_ratio(c_opt, Y, weights)
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_coeffs = c_opt.copy()
    
    # Try more random restarts if time permits
    for i in range(40):
        if time.time() > deadline - safety_margin:
            break
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        # First evaluate this random start
        r_rand = _compute_ratio(c_rand, Y, weights)
        if r_rand < best_ratio:
            best_ratio = r_rand
            best_coeffs = c_rand.copy()
        
        # Run optimization from this random start
        result = minimize(
            objective_and_grad,
            c_rand,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
        
        # If we found a good solution, do a local refinement
        if best_ratio < 1.05:
            result2 = minimize(
                objective_and_grad,
                best_coeffs,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 100}
            )
            if result2.fun < best_ratio:
                c_opt = result2.x / np.linalg.norm(result2.x)
                r_opt = _compute_ratio(c_opt, Y, weights)
                if r_opt < best_ratio:
                    best_ratio = r_opt
                    best_coeffs = c_opt.copy()
    
    # Final refinement from best solution
    if time.time() < deadline - safety_margin:
        result_final = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
        )
        if result_final.fun < best_ratio:
            c_opt = result_final.x / np.linalg.norm(result_final.x)
            r_opt = _compute_ratio(c_opt, Y, weights)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
