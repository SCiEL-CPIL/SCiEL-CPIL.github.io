# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv, gammaln
from scipy.optimize import minimize, differential_evolution


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


def _de_objective(c, Y, weights):
    """Objective for differential evolution: R with normalization."""
    c_norm = np.linalg.norm(c)
    if c_norm < 1e-10:
        return 1e10
    c_normalized = c / c_norm
    return _compute_ratio(c_normalized, Y, weights)


def _polish_local(c_init, Y, weights, deadline, safety_margin=30.0):
    """Polish a starting point using L-BFGS-B on the unit sphere."""
    if time.time() > deadline - safety_margin:
        return c_init, _compute_ratio(c_init, Y, weights)
    
    def objective_and_grad(c):
        if time.time() > deadline - safety_margin:
            R, g = _compute_ratio_and_grad(c_init, Y, weights)
            return R, g
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights)
        dot = np.dot(grad, c_normalized)
        grad_c = (grad - c_normalized * dot) / c_norm
        return R, grad_c
    
    result = minimize(
        objective_and_grad,
        c_init,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 3000, 'ftol': 1e-15, 'gtol': 1e-14, 'maxls': 200}
    )
    
    if result.fun < _compute_ratio(c_init, Y, weights):
        c_opt = result.x / np.linalg.norm(result.x)
        r_opt = _compute_ratio(c_opt, Y, weights)
        return c_opt, r_opt
    return c_init, _compute_ratio(c_init, Y, weights)


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 40.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Initial best: all ones normalized
    best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    best_ratio = _compute_ratio(best_coeffs, Y, weights)
    
    # Phase 1: Differential Evolution for global search
    # DE works in R^21; we normalize inside the objective
    # Bounds: each coefficient in [-1, 1] (normalization handles scale)
    bounds = [(-1.0, 1.0)] * 21
    
    if time.time() < deadline - safety_margin - 120:  # Reserve time for polishing
        de_result = differential_evolution(
            lambda c: _de_objective(c, Y, weights),
            bounds,
            seed=42,
            maxiter=300,
            popsize=25,
            tol=1e-12,
            mutation=(0.3, 1.0),
            recombination=0.7,
            polish=False,  # We'll polish separately
            init='sobol',
            seed=42
        )
        
        if de_result.fun < best_ratio:
            c_de = de_result.x / np.linalg.norm(de_result.x)
            r_de = _compute_ratio(c_de, Y, weights)
            if r_de < best_ratio:
                best_ratio = r_de
                best_coeffs = c_de.copy()
    
    # Phase 2: Polish the best DE result
    if time.time() < deadline - safety_margin:
        best_coeffs, best_ratio = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
    
    # Phase 3: Try a few targeted initial guesses and polish
    rng = np.random.default_rng(123)
    
    # Targeted patterns
    patterns = []
    
    # All cos modes (m=0,2,4,6,8,10)
    c_cos = np.zeros(21)
    for m in [0, 2, 4, 6, 8, 10]:
        c_cos[10 + m] = 1.0
    c_cos /= np.linalg.norm(c_cos)
    patterns.append(c_cos)
    
    # All sin modes
    c_sin = np.zeros(21)
    for m in [-10, -8, -6, -4, -2]:
        c_sin[10 + m] = 1.0
    c_sin /= np.linalg.norm(c_sin)
    patterns.append(c_sin)
    
    # High |m| only
    c_high = np.zeros(21)
    for m in [8, 9, 10, -8, -9, -10]:
        c_high[10 + m] = 1.0
    c_high /= np.linalg.norm(c_high)
    patterns.append(c_high)
    
    # Mixed random with structure
    for _ in range(10):
        c_mix = rng.standard_normal(21)
        # Emphasize high |m|
        for i in range(21):
            m = i - 10
            c_mix[i] *= (1.0 + 0.5 * abs(m) / 10.0)
        c_mix /= np.linalg.norm(c_mix)
        patterns.append(c_mix)
    
    for c_pat in patterns:
        if time.time() > deadline - safety_margin:
            break
        r_pat = _compute_ratio(c_pat, Y, weights)
        if r_pat < best_ratio:
            best_ratio = r_pat
            best_coeffs = c_pat.copy()
        
        # Polish
        c_polished, r_polished = _polish_local(c_pat, Y, weights, deadline, safety_margin)
        if r_polished < best_ratio:
            best_ratio = r_polished
            best_coeffs = c_polished.copy()
    
    # Phase 4: Final refinement
    if time.time() < deadline - safety_margin:
        best_coeffs, best_ratio = _polish_local(best_coeffs, Y, weights, deadline, safety_margin)
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21.0)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
    
    return best_coeffs


# EVOLVE-BLOCK-END
