# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import lpmv, gammaln
import time


def _build_harmonics():
    """
    Build the 21 real orthonormal degree-10 spherical harmonics evaluated
    at 70 Gauss-Legendre nodes in cos(theta) x 130 equally-spaced phi points.
    """
    k = 10
    n_theta = 70
    n_phi = 130
    
    x, w = np.polynomial.legendre.leggauss(n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    phi_weight = 2 * np.pi / n_phi
    weights_2d = np.outer(w, np.full(n_phi, phi_weight))
    
    Y = np.zeros((n_theta, n_phi, 2 * k + 1), dtype=np.float64)
    
    for m in range(-k, k + 1):
        m_idx = m + k
        abs_m = abs(m)
        
        P = lpmv(abs_m, k, x)
        
        log_N = 0.5 * (np.log(21.0) - np.log(4.0 * np.pi) + gammaln(k - abs_m + 1) - gammaln(k + abs_m + 1))
        N = np.exp(log_N)
        
        if m == 0:
            Y[:, :, m_idx] = N * P[:, np.newaxis]
        elif m > 0:
            Y[:, :, m_idx] = np.sqrt(2) * N * P[:, np.newaxis] * np.cos(m * phi)[np.newaxis, :]
        else:
            Y[:, :, m_idx] = np.sqrt(2) * N * P[:, np.newaxis] * np.sin(abs_m * phi)[np.newaxis, :]
    
    return Y, weights_2d


def _objective_and_grad(coeffs, Y, weights):
    """Compute R = ||f||_4 / ||f||_2 and its gradient."""
    f = np.tensordot(coeffs, Y, axes=([0], [2]))
    
    f2 = f * f
    norm2_sq = np.sum(f2 * weights)
    
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    
    f4 = f2 * f2
    norm4_4 = np.sum(f4 * weights)
    
    if norm4_4 < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm4 = norm4_4 ** 0.25
    R = norm4 / norm2
    
    f3_w = f * f2 * weights
    f_w = f * weights
    
    sum_f3_Y = np.tensordot(f3_w, Y, axes=([0, 1], [0, 1]))
    sum_f_Y = np.tensordot(f_w, Y, axes=([0, 1], [0, 1]))
    
    d_norm4_dc = (norm4_4 ** (-0.75)) * sum_f3_Y
    d_norm2_dc = (norm2_sq ** (-0.5)) * sum_f_Y
    
    grad = (1.0 / norm2) * d_norm4_dc - (norm4 / (norm2 * norm2)) * d_norm2_dc
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find coefficients minimizing ||f||_4/||f||_2 for degree-10 spherical harmonics.
    """
    k = 10
    n_coeffs = 2 * k + 1
    
    start_time = time.time()
    deadline = start_time + 570  # safety margin before 630s kill
    
    Y, weights = _build_harmonics()
    
    best_coeffs = None
    best_R = np.inf
    
    inits = []
    
    # m=0 only
    c0 = np.zeros(n_coeffs)
    c0[k] = 1.0
    inits.append(c0)
    
    # all ones
    inits.append(np.ones(n_coeffs))
    
    # random inits
    np.random.seed(42)
    for _ in range(5):
        inits.append(np.random.randn(n_coeffs))
    
    for c_init in inits:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        R_init, _ = _objective_and_grad(c_init, Y, weights)
        if R_init < best_R:
            best_R = R_init
            best_coeffs = c_init.copy()
        
        def obj_grad(c):
            return _objective_and_grad(c, Y, weights)
        
        try:
            result = minimize(
                obj_grad, c_init, method='L-BFGS-B', jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            if result.fun < best_R:
                best_R = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time > deadline - 30:
            break
    
    if best_coeffs is None:
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0
    
    if not np.all(np.isfinite(best_coeffs)):
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = np.zeros(n_coeffs)
        best_coeffs[k] = 1.0
    elif max_abs > 1e5:
        best_coeffs = best_coeffs / max_abs * 1e5
    
    return best_coeffs


# EVOLVE-BLOCK-END
