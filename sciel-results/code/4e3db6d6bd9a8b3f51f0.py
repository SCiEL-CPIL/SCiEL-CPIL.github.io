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


def _build_fourth_power_kernel(Y, weights):
    """
    Build the matrix K such that ||f||_4^4 = sum_{i,j} c_i^2 K_{ij} c_j^2
    where f = sum c_i Y_i and K_{ij} = sum_p w_p Y_i(p)^2 Y_j(p)^2.
    """
    n_modes = Y.shape[0]
    n_pts = Y.shape[1] * Y.shape[2]
    
    Y_flat = Y.reshape(n_modes, n_pts)
    w_flat = weights.reshape(n_pts)
    
    Y_sq_w = Y_flat * Y_flat * w_flat[np.newaxis, :]
    K = Y_sq_w @ Y_flat.T
    
    return K


def _compute_ratio_and_grad(coeffs, Y, weights, K):
    """
    Compute R = ||f||_4 / ||f||_2 and its gradient using the spectral kernel.
    coeffs: shape (21,)
    Y: shape (21, n_theta, n_phi)
    weights: shape (n_theta, n_phi)
    K: shape (21, 21) - fourth power kernel
    """
    n_modes = len(coeffs)
    c = coeffs
    
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    norm2_sq = np.sum(weights * f2)
    
    c_sq = c * c
    norm4_4 = c_sq @ K @ c_sq
    
    if norm2_sq <= 1e-15 or norm4_4 <= 1e-15:
        return np.inf, np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    R = norm4 / norm2
    
    Kc_sq = K @ c_sq
    dnorm4_4_dc = 2.0 * c * Kc_sq
    
    dnorm4_dc = 0.25 * norm4_4 ** (-0.75) * dnorm4_4_dc
    
    weighted_f = weights * f
    dnorm2_sq_dc = 2.0 * np.tensordot(weighted_f, Y, axes=([0, 1], [1, 2]))
    
    dnorm2_dc = 0.5 / norm2 * dnorm2_sq_dc
    
    dR_dc = dnorm4_dc / norm2 - norm4 / norm2_sq * dnorm2_dc
    
    return R, dR_dc


def _compute_ratio_only(coeffs, Y, weights, K):
    """Compute only the ratio R, without gradient, using the kernel."""
    c = coeffs
    f = np.tensordot(c, Y, axes=([0], [0]))
    f2 = f * f
    norm2_sq = np.sum(weights * f2)
    
    c_sq = c * c
    norm4_4 = c_sq @ K @ c_sq
    
    if norm2_sq <= 1e-15 or norm4_4 <= 1e-15:
        return np.inf
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    return norm4 / norm2


def _build_cg_tensor(Y, weights):
    """
    Build the 3D tensor T[i,j,k] = sum_p w_p Y_i(p) Y_j(p) Y_k(p)
    such that ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l T[i,j,k] T[l,?,?]
    Actually for real harmonics: ||f||_4^4 = sum_{i,j,k,l} c_i c_j c_k c_l * (sum_p w_p Y_i Y_j Y_k Y_l)
    We can factor: let S[i,j,k] = sum_p w_p Y_i(p) Y_j(p) Y_k(p)
    Then ||f||_4^4 = sum_{i,j,k} c_i c_j c_k c_l * ... 
    Actually simpler: use K[i,j] = sum_p w_p Y_i(p)^2 Y_j(p)^2 and compute c^T K c on c^2.
    But for gradient via tensor: d||f||_4^4/dc_m = 4 * sum_{i,j,k} c_i c_j c_k * S[m,i,j,k]
    where S[m,i,j,k] = sum_p w_p Y_m(p) Y_i(p) Y_j(p) Y_k(p)
    
    Alternative: precompute K[i,j] = sum_p w_p Y_i(p)^2 Y_j(p)^2
    Then ||f||_4^4 = (c^2)^T K (c^2)
    d||f||_4^4/dc_m = 4 * c_m * (K c^2)_m
    """
    n_modes = Y.shape[0]
    n_pts = Y.shape[1] * Y.shape[2]
    
    Y_flat = Y.reshape(n_modes, n_pts)
    w_flat = weights.reshape(n_pts)
    
    # K[i,j] = sum_p w_p Y_i(p)^2 Y_j(p)^2
    Y_sq = Y_flat * Y_flat
    Y_sq_w = Y_sq * w_flat[np.newaxis, :]
    K = Y_sq_w @ Y_flat.T
    
    # For gradient: d(norm4_4)/dc_m = 4 * c_m * (K @ c_sq)_m
    # This is already captured in the existing gradient formula.
    # The "tensor" approach is essentially what K already provides.
    return K


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing R = ||f||_4 / ||f||_2 for degree-10 spherical harmonic.
    Returns 21 real coefficients ordered m = -10..10.
    """
    deadline = time.time() + 600.0
    safety_margin = 45.0
    
    # Build quadrature grid
    cos_theta, phi, weights = _build_quadrature()
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, k=10)
    
    # Precompute the fourth-power kernel for fast L4 norm evaluation
    K = _build_fourth_power_kernel(Y, weights)
    
    # Precompute Gram matrix (should be ~identity for orthonormal basis)
    Y_flat = Y.reshape(21, -1)
    w_flat = weights.reshape(-1)
    G = (Y_flat * w_flat[np.newaxis, :]) @ Y_flat.T
    
    best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
    best_ratio, _ = _compute_ratio_and_grad(best_coeffs, Y, weights, K)
    
    # Generate structured initial guesses
    rng = np.random.default_rng(42)
    initial_guesses = []
    
    # Uniform
    c0 = np.ones(21) / np.sqrt(21)
    initial_guesses.append(c0)
    
    # Single modes
    for idx in range(21):
        c_single = np.zeros(21)
        c_single[idx] = 1.0
        initial_guesses.append(c_single)
    
    # Pairs of symmetric modes
    for m_val in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        c_pair = np.zeros(21)
        c_pair[10 + m_val] = 1.0
        c_pair[10 - m_val] = 1.0
        c_pair /= np.linalg.norm(c_pair)
        initial_guesses.append(c_pair)
        
        c_pair2 = np.zeros(21)
        c_pair2[10 + m_val] = 1.0
        c_pair2[10 - m_val] = -1.0
        c_pair2 /= np.linalg.norm(c_pair2)
        initial_guesses.append(c_pair2)
    
    # m=0 with neighbors
    for m_val in [1, 2, 3]:
        c_mix = np.zeros(21)
        c_mix[10] = 1.0
        c_mix[10 + m_val] = 1.0
        c_mix[10 - m_val] = 1.0
        c_mix /= np.linalg.norm(c_mix)
        initial_guesses.append(c_mix)
    
    # Random
    for i in range(60):
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        initial_guesses.append(c_rand)
    
    for c_init in initial_guesses:
        r = _compute_ratio_only(c_init, Y, weights, K)
        if r < best_ratio:
            best_ratio = r
            best_coeffs = c_init.copy()
    
    # Optimization with trust-region method
    def objective_and_grad(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10, np.zeros_like(c)
        
        c_normalized = c / c_norm
        R, grad = _compute_ratio_and_grad(c_normalized, Y, weights, K)
        
        # Project gradient onto tangent space of sphere
        dot = np.dot(grad, c_normalized)
        grad_c = (grad - c_normalized * dot) / c_norm
        
        return R, grad_c
    
    # Phase 1: Differential Evolution for global search
    def de_objective(c):
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            return 1e10
        c_normalized = c / c_norm
        return _compute_ratio_only(c_normalized, Y, weights, K)
    
    bounds = [(-1.0, 1.0)] * 21
    
    if time.time() < deadline - safety_margin - 120:
        de_result = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=80,
            popsize=20,
            tol=1e-12,
            polish=False,
            init='sobol'
        )
        
        if de_result.fun < best_ratio:
            c_de = de_result.x / np.linalg.norm(de_result.x)
            r_de = _compute_ratio_only(c_de, Y, weights, K)
            if r_de < best_ratio:
                best_ratio = r_de
                best_coeffs = c_de.copy()
    
    # Phase 2: Trust-region reflective polishing
    if time.time() < deadline - safety_margin:
        result = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='trust-reflect',
            options={'maxiter': 5000, 'gtol': 1e-14, 'xtol': 1e-14, 'disp': False}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Phase 3: L-BFGS-B from multiple restarts
    for i in range(100):
        if time.time() > deadline - safety_margin:
            break
            
        c_rand = rng.standard_normal(21)
        c_rand /= np.linalg.norm(c_rand)
        
        max_iter = 2000 if i < 30 else 1500
        
        result = minimize(
            objective_and_grad,
            c_rand,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 50}
        )
        
        if result.fun < best_ratio:
            c_opt = result.x / np.linalg.norm(result.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
                
    # Phase 4: High-precision trust-region polish
    if time.time() < deadline - safety_margin:
        result_final = minimize(
            objective_and_grad,
            best_coeffs,
            jac=True,
            method='trust-reflect',
            options={'maxiter': 10000, 'gtol': 1e-16, 'xtol': 1e-16, 'disp': False}
        )
        if result_final.fun < best_ratio:
            c_opt = result_final.x / np.linalg.norm(result_final.x)
            r_opt = _compute_ratio_only(c_opt, Y, weights, K)
            if r_opt < best_ratio:
                best_ratio = r_opt
                best_coeffs = c_opt.copy()
    
    # Phase 5: Additional DE runs if time allows
    if time.time() < deadline - safety_margin - 90:
        for seed in [123, 456, 789, 101, 202]:
            if time.time() > deadline - safety_margin:
                break
            try:
                de_result2 = differential_evolution(
                    de_objective,
                    bounds,
                    seed=seed,
                    maxiter=50,
                    popsize=15,
                    tol=1e-12,
                    polish=False,
                    init='sobol'
                )
                if de_result2.fun < best_ratio:
                    c_de2 = de_result2.x / np.linalg.norm(de_result2.x)
                    r_de2 = _compute_ratio_only(c_de2, Y, weights, K)
                    if r_de2 < best_ratio:
                        best_ratio = r_de2
                        best_coeffs = c_de2.copy()
                        
                        if time.time() < deadline - safety_margin:
                            result_polish = minimize(
                                objective_and_grad,
                                best_coeffs,
                                jac=True,
                                method='trust-reflect',
                                options={'maxiter': 5000, 'gtol': 1e-15, 'xtol': 1e-15, 'disp': False}
                            )
                            if result_polish.fun < best_ratio:
                                c_opt = result_polish.x / np.linalg.norm(result_polish.x)
                                r_opt = _compute_ratio_only(c_opt, Y, weights, K)
                                if r_opt < best_ratio:
                                    best_ratio = r_opt
                                    best_coeffs = c_opt.copy()
            except Exception:
                pass
    
    # Ensure output is valid
    best_coeffs = best_coeffs.astype(np.float64)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21, dtype=np.float64) / np.sqrt(21)
        
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        target = 1.0
        best_coeffs = best_coeffs * (target / max_abs)
        
    return best_coeffs


# EVOLVE-BLOCK-END
