# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
from scipy.optimize import minimize, differential_evolution


def _precompute_basis():
    """
    Precompute the quadrature grid and spherical harmonic basis values.
    
    Returns:
        w_theta: (70,) Gauss-Legendre weights
        Y_matrix: (70, 130, 21) Y_m(x_i, phi_j) for m=-10..10
        N_phi: int, number of phi points
    """
    k = 10
    N_theta = 70
    N_phi = 130
    
    # Gauss-Legendre nodes and weights on [-1, 1]
    theta_grid, w_theta = np.polynomial.legendre.leggauss(N_theta)
    
    # Equally spaced phi
    phi_grid = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
    
    # Precompute Y_m(x_i, phi_j) for all m, i, j
    Y_matrix = np.zeros((N_theta, N_phi, 2 * k + 1), dtype=np.float64)
    
    for m_idx, m in enumerate(range(-k, k + 1)):
        abs_m = abs(m)
        
        # Condon-Shortley phase is included in scipy's lpmv
        P = lpmv(abs_m, k, theta_grid)  # (N_theta,)
        
        # Normalization: N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        if abs_m == 0:
            fact_ratio = 1.0
        else:
            prod = 1.0
            for j in range(10 - abs_m + 1, 10 + abs_m + 1):
                prod *= j
            fact_ratio = 1.0 / prod
        
        N_m = np.sqrt(21.0 / (4.0 * np.pi) * fact_ratio)
        
        if m == 0:
            Y_vals = N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * np.ones((1, N_phi))[np.newaxis, :]
        elif m > 0:
            cos_term = np.cos(m * phi_grid)  # (N_phi,)
            Y_vals = np.sqrt(2.0) * N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * cos_term[np.newaxis, :]
        else:
            sin_term = np.sin(abs_m * phi_grid)  # (N_phi,)
            Y_vals = np.sqrt(2.0) * N_m * P  # (N_theta,)
            Y_matrix[:, :, m_idx] = Y_vals[:, np.newaxis] * sin_term[np.newaxis, :]
    
    return w_theta, Y_matrix, N_phi


def _compute_R(c, w_theta, Y_matrix, N_phi):
    """
    Compute R = ||f||_4 / ||f||_2 using precomputed data.
    """
    f = np.tensordot(c, Y_matrix, axes=([0], [2]))  # (N_theta, N_phi)
    
    factor = 2.0 * np.pi / N_phi
    
    f2 = f ** 2
    f4 = f ** 4
    
    sum_j_f2 = np.sum(f2, axis=1)
    sum_j_f4 = np.sum(f4, axis=1)
    
    integral_f2 = np.sum(w_theta * sum_j_f2)
    integral_f4 = np.sum(w_theta * sum_j_f4)
    
    norm2_sq = factor * integral_f2
    norm4_4 = factor * integral_f4
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    R = norm4 / norm2
    return R


def _compute_grad_R(c, w_theta, Y_matrix, N_phi):
    """
    Compute the gradient of R with respect to c.
    """
    f = np.tensordot(c, Y_matrix, axes=([0], [2]))  # (N_theta, N_phi)
    
    factor = 2.0 * np.pi / N_phi
    
    f2 = f ** 2
    f3 = f ** 3
    f4 = f ** 4
    
    # integral f Y_m dOmega
    temp_fY = f[:, :, np.newaxis] * Y_matrix  # (N_theta, N_phi, 21)
    sum_j_fY = np.sum(temp_fY, axis=1)  # (N_theta, 21)
    integral_f_Ym = factor * np.sum(w_theta[:, np.newaxis] * sum_j_fY, axis=0)  # (21,)
    
    # integral f^3 Y_m dOmega
    temp_f3Y = f3[:, :, np.newaxis] * Y_matrix  # (N_theta, N_phi, 21)
    sum_j_f3Y = np.sum(temp_f3Y, axis=1)  # (N_theta, 21)
    integral_f3_Ym = factor * np.sum(w_theta[:, np.newaxis] * sum_j_f3Y, axis=0)  # (21,)
    
    # Compute norms
    sum_j_f2 = np.sum(f2, axis=1)
    sum_j_f4 = np.sum(f4, axis=1)
    integral_f2 = np.sum(w_theta * sum_j_f2)
    integral_f4 = np.sum(w_theta * sum_j_f4)
    
    norm2_sq = factor * integral_f2
    norm4_4 = factor * integral_f4
    
    if norm2_sq <= 0 or norm4_4 <= 0:
        return np.zeros_like(c)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))
    
    coeff1 = 1.0 / (norm2 * norm4**3)
    coeff2 = norm4 / (norm2**3)
    
    grad = coeff1 * integral_f3_Ym - coeff2 * integral_f_Ym
    
    return grad


def _refine_with_lbfgs(c0, w_theta, Y_matrix, N_phi, max_iter=5000):
    """
    Refine coefficients using L-BFGS-B optimizer.
    """
    def fun(c):
        return _compute_R(c, w_theta, Y_matrix, N_phi)
    
    def jac(c):
        return _compute_grad_R(c, w_theta, Y_matrix, N_phi)
    
    # Normalize initial guess
    c0 = c0.copy()
    c0_norm = np.linalg.norm(c0)
    if c0_norm > 0:
        c0 = c0 / c0_norm
    
    try:
        result = minimize(fun, c0, jac=jac, method='L-BFGS-B', 
                          options={'maxiter': max_iter, 'gtol': 1e-12, 'ftol': 1e-14})
        return result.x, result.fun
    except Exception:
        return c0, fun(c0)


def _refine_with_nelder_mead(c0, w_theta, Y_matrix, N_phi, maxiter=2000):
    """
    Refine coefficients using Nelder-Mead (derivative-free).
    """
    def fun(c):
        return _compute_R(c, w_theta, Y_matrix, N_phi)
    
    c0 = c0.copy()
    c0_norm = np.linalg.norm(c0)
    if c0_norm > 0:
        c0 = c0 / c0_norm
    
    try:
        result = minimize(fun, c0, method='Nelder-Mead',
                          options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14,
                                   'adaptive': True})
        return result.x, result.fun
    except Exception:
        return c0, fun(c0)


def _differential_evolution_search(w_theta, Y_matrix, N_phi, max_iter=150, popsize=20):
    """
    Use differential evolution for global search.
    """
    def fun(c):
        return _compute_R(c, w_theta, Y_matrix, N_phi)
    
    # Bounds: each coefficient in [-1, 1]
    bounds = [(-1.0, 1.0)] * 21
    
    try:
        result = differential_evolution(fun, bounds, maxiter=max_iter, 
                                         popsize=popsize, seed=42, 
                                         tol=1e-12, polish=True,
                                         mutation=(0.5, 1.5))
        return result.x, result.fun
    except Exception:
        return None, np.inf


def _generate_diverse_starts(n_random=15):
    """
    Generate diverse starting points for optimization.
    """
    starts = []
    
    # All ones
    c1 = np.ones(21) / np.sqrt(21.0)
    starts.append(c1)
    
    # Alternating signs
    c2 = np.array([(-1)**(i) for i in range(21)], dtype=float)
    c2 /= np.linalg.norm(c2)
    starts.append(c2)
    
    # Only m=0
    c3 = np.zeros(21)
    c3[10] = 1.0
    starts.append(c3)
    
    # Only m=10
    c4 = np.zeros(21)
    c4[20] = 1.0
    starts.append(c4)
    
    # Only m=-10
    c5 = np.zeros(21)
    c5[0] = 1.0
    starts.append(c5)
    
    # Only m=5
    c6 = np.zeros(21)
    c6[15] = 1.0
    starts.append(c6)
    
    # Only m=-5
    c7 = np.zeros(21)
    c7[5] = 1.0
    starts.append(c7)
    
    # Symmetric pair m=±10
    c8 = np.zeros(21)
    c8[0] = 1.0
    c8[20] = 1.0
    c8 /= np.linalg.norm(c8)
    starts.append(c8)
    
    # Symmetric pair m=±5
    c9 = np.zeros(21)
    c9[5] = 1.0
    c9[15] = 1.0
    c9 /= np.linalg.norm(c9)
    starts.append(c9)
    
    # m=0 and m=±10
    c10 = np.zeros(21)
    c10[0] = 1.0
    c10[10] = 1.0
    c10[20] = 1.0
    c10 /= np.linalg.norm(c10)
    starts.append(c10)
    
    # m=0 and m=±5
    c11 = np.zeros(21)
    c11[5] = 1.0
    c11[10] = 1.0
    c11[15] = 1.0
    c11 /= np.linalg.norm(c11)
    starts.append(c11)
    
    # Random starts with different seeds
    np.random.seed(12345)
    for i in range(n_random):
        c_rand = np.random.randn(21)
        c_rand /= np.linalg.norm(c_rand)
        starts.append(c_rand)
    
    return starts


def _multi_start_optimization(w_theta, Y_matrix, N_phi, n_random_starts=15):
    """
    Multi-start optimization: combine deterministic starts, random starts,
    and differential evolution.
    """
    best_c = None
    best_R = np.inf
    
    # Generate diverse starts
    starts = _generate_diverse_starts(n_random_starts)
    
    # Refine each start with L-BFGS-B
    for c_start in starts:
        c_refined, R_refined = _refine_with_lbfgs(c_start, w_theta, Y_matrix, N_phi)
        if R_refined < best_R:
            best_R = R_refined
            best_c = c_refined.copy()
    
    # Try Nelder-Mead on the best result so far
    if best_c is not None:
        c_nm, R_nm = _refine_with_nelder_mead(best_c, w_theta, Y_matrix, N_phi)
        if R_nm < best_R:
            best_R = R_nm
            best_c = c_nm.copy()
    
    # Try differential evolution for global search
    c_de, R_de = _differential_evolution_search(w_theta, Y_matrix, N_phi)
    if c_de is not None and R_de < best_R:
        best_R = R_de
        best_c = c_de.copy()
        # Refine the DE result with both methods
        c_refined, R_refined = _refine_with_lbfgs(best_c, w_theta, Y_matrix, N_phi)
        if R_refined < best_R:
            best_R = R_refined
            best_c = c_refined.copy()
        c_nm2, R_nm2 = _refine_with_nelder_mead(best_c, w_theta, Y_matrix, N_phi)
        if R_nm2 < best_R:
            best_R = R_nm2
            best_c = c_nm2.copy()
    
    # Additional refinement pass on the best result
    if best_c is not None:
        c_final, R_final = _refine_with_lbfgs(best_c, w_theta, Y_matrix, N_phi, max_iter=10000)
        if R_final < best_R:
            best_R = R_final
            best_c = c_final.copy()
    
    return best_c, best_R


def rudin_problem10() -> np.ndarray:
    """
    Rudin problem for polynomials on the 2-sphere. Find the real coefficients
    (c_m) for m = -k..k of a real degree-k spherical harmonic

        f = sum_{m=-k}^{k} c_m Y_k^m

    on the round unit sphere S^2 that MINIMIZE the norm ratio

        ||f||_{L^4(S^2)} / ||f||_{L^2(S^2)}

    (norms taken with the standard surface measure dOmega of total mass 4*pi).
    Here the degree is fixed to k = 10, so the returned vector has length
    2*k + 1 = 21.

    Returns:
        coeffs: np.ndarray of shape (21,) with the real coefficients
            c_m, ordered m = -10, -9, ..., 0, ..., 9, 10.
    """
    # Precompute basis
    w_theta, Y_matrix, N_phi = _precompute_basis()
    
    # Multi-start optimization
    best_c, best_R = _multi_start_optimization(w_theta, Y_matrix, N_phi, n_random_starts=15)
    
    if best_c is None:
        best_c = np.ones(21) / np.sqrt(21.0)
    
    # Scale the result to be within valid range
    c = best_c.copy()
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    # Final validation
    if len(c) != 21:
        c = np.ones(21)
    if np.any(np.isnan(c)) or np.any(np.isinf(c)):
        c = np.ones(21)
    max_abs = np.max(np.abs(c))
    if max_abs < 1e-3:
        c = c * (1.0 / max_abs)
    elif max_abs > 1e5:
        c = c * (1e5 / max_abs)
    
    return c


# EVOLVE-BLOCK-END
