# EVOLVE-BLOCK-START
import numpy as np
from scipy.special import lpmv
import time
import warnings


def _build_quadrature():
    """Build the 70-point Gauss-Legendre quadrature in cos(theta) and 130-point phi grid."""
    k = 10
    n_theta = 70
    n_phi = 130
    
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(n_theta)
    
    # x = cos(theta), theta in [0, pi]
    # dOmega = sin(theta) dtheta dphi = d(cos(theta)) dphi
    # So the integral over sphere is sum_i w_i sum_j f(theta_i, phi_j) * (2*pi/n_phi)
    # Actually: integral = (2*pi/n_phi) * sum_i w_i * sum_j f(x_i, phi_j)
    
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    
    return x, w, phi


def _compute_f(c, x, phi):
    """
    Compute f = sum_m c_m Y_m on the grid.
    x = cos(theta), phi = azimuthal angle.
    Y_0 = N_0 P_10^0(x)
    Y_m = sqrt(2) N_m P_10^m(x) cos(m*phi) for m > 0
    Y_m = sqrt(2) N_|m| P_10^|m|(x) sin(|m|*phi) for m < 0
    N_m = sqrt(21/(4*pi) * (10-|m|)!/(10+|m|)!)
    """
    k = 10
    n_theta = len(x)
    n_phi = len(phi)
    
    # Precompute P_10^m(x) for m = 0..10
    # lpmv(m, l, x) gives the associated Legendre function with Condon-Shortley phase
    P = np.zeros((11, n_theta), dtype=np.float64)
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
    
    # Normalization constants
    N = np.zeros(11, dtype=np.float64)
    for m in range(11):
        # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
        # Use log-gamma for numerical stability
        from math import lgamma, pi
        log_N = 0.5 * (np.log(21.0 / (4.0 * pi)) + lgamma(k - m + 1) - lgamma(k + m + 1))
        N[m] = np.exp(log_N)
    
    # Build f on the grid
    f = np.zeros((n_theta, n_phi), dtype=np.float64)
    
    # m = 0
    f += c[10] * N[0] * P[0, :][:, None]  # broadcast over phi
    
    # m > 0: Y_m = sqrt(2) N_m P^m cos(m*phi)
    for m in range(1, 11):
        coeff = np.sqrt(2.0) * N[m] * P[m, :]
        f += c[10 + m] * coeff[:, None] * np.cos(m * phi)[None, :]
    
    # m < 0: Y_m = sqrt(2) N_|m| P^|m| sin(|m|*phi)
    for m in range(1, 11):
        coeff = np.sqrt(2.0) * N[m] * P[m, :]
        f += c[10 - m] * coeff[:, None] * np.sin(m * phi)[None, :]
    
    return f


def _compute_ratio(c):
    """Compute R = ||f||_4 / ||f||_2 using quadrature."""
    x, w, phi = _QUAD
    n_phi = len(phi)
    dphi = 2 * np.pi / n_phi
    
    f = _compute_f(c, x, phi)
    
    # Integral of |f|^p over sphere = (2*pi/n_phi) * sum_i w_i * sum_j |f_ij|^p
    f2 = f * f
    f4 = f2 * f2
    
    # ||f||_2^2 = integral of f^2 = (2*pi/n_phi) * sum_i w_i * sum_j f_ij^2
    norm2_sq = (2.0 * np.pi / n_phi) * np.sum(w[:, None] * f2, axis=(0, 1))
    # Actually: sum_i w_i * sum_j f_ij^2, then multiply by (2*pi/n_phi)
    # But the quadrature: integral = sum_i w_i * (1/(2*pi)) * integral over phi
    # With uniform phi: integral over phi ≈ (2*pi/n_phi) * sum_j
    # So integral = sum_i w_i * (2*pi/n_phi) * sum_j f_ij^p
    
    # Let's be more careful:
    # integral_{S^2} g dOmega = integral_{-1}^{1} integral_0^{2pi} g(x, phi) dphi dx
    # ≈ sum_i w_i * sum_j g(x_i, phi_j) * (2*pi/n_phi)
    
    integral_f2 = (2.0 * np.pi / n_phi) * np.sum(w[:, None] * f2, axis=(0, 1))
    integral_f4 = (2.0 * np.pi / n_phi) * np.sum(w[:, None] * f4, axis=(0, 1))
    
    norm2 = np.sqrt(integral_f2)
    norm4 = np.sqrt(integral_f4)  # Wait, ||f||_4 = (integral |f|^4)^{1/4}
    
    # Actually ||f||_4 = (integral |f|^4 dOmega)^{1/4}
    # And ||f||_2 = (integral |f|^2 dOmega)^{1/2}
    
    norm2 = np.sqrt(integral_f2)
    norm4 = np.power(integral_f4, 0.25)
    
    if norm2 < 1e-15:
        return np.inf
    
    return norm4 / norm2


# Precompute quadrature
_QUAD = _build_quadrature()


def _compute_ratio_and_grad(c):
    """Compute R and its gradient with respect to c."""
    x, w, phi = _QUAD
    n_phi = len(phi)
    n_theta = len(x)
    dphi = 2.0 * np.pi / n_phi
    
    k = 10
    
    # Precompute P and N
    from math import lgamma, pi
    P = np.zeros((11, n_theta), dtype=np.float64)
    for m in range(11):
        P[m, :] = lpmv(m, k, x)
    
    N = np.zeros(11, dtype=np.float64)
    for m in range(11):
        log_N = 0.5 * (np.log(21.0 / (4.0 * pi)) + lgamma(k - m + 1) - lgamma(k + m + 1))
        N[m] = np.exp(log_N)
    
    # Build f and its derivatives
    f = np.zeros((n_theta, n_phi), dtype=np.float64)
    
    # Y components: Y_j for j=0..20 (j=10 is m=0, j=10+m is m>0, j=10-m is m<0)
    # Y_0 (m=0): N_0 P_10^0
    Y = np.zeros((21, n_theta, n_phi), dtype=np.float64)
    Y[10, :, :] = N[0] * P[0, :][:, None]
    
    for m in range(1, 11):
        Y[10 + m, :, :] = np.sqrt(2.0) * N[m] * P[m, :][:, None] * np.cos(m * phi)[None, :]
        Y[10 - m, :, :] = np.sqrt(2.0) * N[m] * P[m, :][:, None] * np.sin(m * phi)[None, :]
    
    f = np.tensordot(c, Y, axes=(0, 0))  # shape (n_theta, n_phi)
    
    f2 = f * f
    f4 = f2 * f2
    
    integral_f2 = dphi * np.sum(w[:, None] * f2, axis=(0, 1))
    integral_f4 = dphi * np.sum(w[:, None] * f4, axis=(0, 1))
    
    norm2_sq = integral_f2
    norm4_4th = integral_f4
    
    if norm2_sq < 1e-15:
        return np.inf, np.zeros(21)
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.power(norm4_4th, 0.25)
    R = norm4 / norm2
    
    # Gradient: dR/dc_j
    # R = (integral_f4)^{1/4} / (integral_f2)^{1/2}
    # dR/dj = (1/4)(integral_f4)^{-3/4} * d(integral_f4)/dj / (integral_f2)^{1/2}
    #        - (integral_f4)^{1/4} * (1/2)(integral_f2)^{-3/2} * d(integral_f2)/dj
    #
    # d(integral_f2)/dj = dphi * sum w * 2*f*Y_j
    # d(integral_f4)/dj = dphi * sum w * 4*f^3*Y_j
    
    d_int_f2 = dphi * np.sum(w[:, None] * 2.0 * f * Y, axis=(1, 2))  # shape (21,)
    d_int_f4 = dphi * np.sum(w[:, None] * 4.0 * f2 * f * Y, axis=(1, 2))  # shape (21,)
    
    grad = (0.25 * np.power(norm4_4th, -0.75) * d_int_f4 / norm2
            - norm4 * 0.5 * np.power(norm2_sq, -1.5) * d_int_f2)
    
    return R, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients c_m for m = -10..10 that minimize R = ||f||_4 / ||f||_2.
    """
    k = 10
    n = 2 * k + 1  # 21
    
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    best_c = np.ones(n, dtype=np.float64)
    best_R = _compute_ratio(best_c)
    
    # Try multiple random initializations
    np.random.seed(42)
    
    # Strategy: use scipy.optimize.minimize with L-BFGS-B or SLSQP
    # Since the objective is homogeneous (scaling c doesn't change R),
    # we can fix ||c||_2 = 1 to remove the gauge freedom.
    
    from scipy.optimize import minimize
    
    def objective(c):
        return _compute_ratio(c)
    
    def objective_and_grad(c):
        return _compute_ratio_and_grad(c)
    
    # Try several random starts on the unit sphere
    n_starts = 20
    for s in range(n_starts):
        if time.time() > deadline - 30.0:
            break
        
        # Random start on unit sphere
        c0 = np.random.randn(n)
        c0 /= np.linalg.norm(c0)
        
        try:
            result = minimize(
                objective_and_grad,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
            )
            
            c_opt = result.x
            R_opt = objective(c_opt)
            
            if np.isfinite(R_opt) and R_opt < best_R:
                best_R = R_opt
                best_c = c_opt.copy()
        except Exception as e:
            warnings.warn(f"Optimization failed for start {s}: {e}")
            continue
    
    # Also try a few more targeted starts
    # Start with only one coefficient nonzero
    for idx in range(n):
        if time.time() > deadline - 30.0:
            break
        c0 = np.zeros(n)
        c0[idx] = 1.0
        try:
            result = minimize(
                objective_and_grad,
                c0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 200}
            )
            c_opt = result.x
            R_opt = objective(c_opt)
            if np.isfinite(R_opt) and R_opt < best_R:
                best_R = R_opt
                best_c = c_opt.copy()
        except Exception:
            continue
    
    # Final polish with SLSQP if time permits
    if time.time() < deadline - 60.0:
        try:
            result = minimize(
                objective_and_grad,
                best_c,
                method='SLSQP',
                jac=True,
                options={'maxiter': 200, 'ftol': 1e-14, 'eps': 1e-8}
            )
            c_opt = result.x
            R_opt = objective(c_opt)
            if np.isfinite(R_opt) and R_opt < best_R:
                best_R = R_opt
                best_c = c_opt.copy()
        except Exception:
            pass
    
    # Validate
    if best_c.size != n:
        best_c = np.ones(n, dtype=np.float64)
    
    if not np.all(np.isfinite(best_c)):
        best_c = np.ones(n, dtype=np.float64)
    
    max_abs = np.max(np.abs(best_c))
    if max_abs < 1e-3 or max_abs > 1e5:
        # Rescale
        if max_abs > 0:
            best_c = best_c / max_abs * 1.0
        else:
            best_c = np.ones(n, dtype=np.float64)
    
    return best_c


# EVOLVE-BLOCK-END
