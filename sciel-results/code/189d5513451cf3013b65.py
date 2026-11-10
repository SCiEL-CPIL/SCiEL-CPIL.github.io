# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

def _build_quadrature():
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    # Gauss-Legendre nodes and weights in [-1, 1]
    x, w = np.polynomial.legendre.leggauss(70)
    # x = cos(theta), w are weights for integral over cos(theta)
    # We need surface measure: dOmega = dphi * d(cos theta)
    # Total integral of 1 over sphere = 4*pi
    # Integral over cos(theta) in [-1,1] is 2, over phi in [0, 2pi) is 2pi
    # So weight for each (theta, phi) point: w_i * (2*pi/130)
    n_phi = 130
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    # Weights: for each theta point i, phi point j: w_i * (2*pi/n_phi)
    w_theta = w  # shape (70,)
    w_phi = 2 * np.pi / n_phi  # scalar
    return x, w_theta, phi, w_phi


def _spherical_harmonics(x, phi, degree=10):
    """
    Compute real orthonormal spherical harmonics of given degree.
    Returns array of shape (n_theta, n_phi, 2*degree+1) where index m corresponds to m-10.
    
    Y_0 = N_0 * P_10^0(cos theta)
    Y_m = sqrt(2) * N_m * P_10^m(cos theta) * cos(m*phi) for m > 0
    Y_m = sqrt(2) * N_|m| * P_10^|m|(cos theta) * sin(|m|*phi) for m < 0
    
    where N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    and P_10^m uses Condon-Shortley phase (scipy's lpmv includes it).
    """
    k = degree
    n_theta = len(x)
    n_phi = len(phi)
    
    # Precompute normalization constants
    # N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    # For m = 0, 1, ..., 10
    norm_constants = np.zeros(k + 1)
    for m in range(k + 1):
        # (10-m)! / (10+m)!
        ratio = 1.0
        for j in range(10 - m + 1, 10 + m + 1):
            ratio /= j
        norm_constants[m] = np.sqrt(21.0 / (4.0 * np.pi) * ratio)
    
    # Precompute associated Legendre polynomials P_10^m(x) for m = 0..10
    # x has shape (n_theta,)
    P = np.zeros((n_theta, k + 1))
    for m in range(k + 1):
        P[:, m] = lpmv(m, k, x)
    
    # Build the harmonic values
    # Shape: (n_theta, n_phi, 2*k+1)
    Y = np.zeros((n_theta, n_phi, 2 * k + 1))
    
    # For m = 0
    Y[:, :, k] = norm_constants[0] * P[:, 0][:, np.newaxis]
    
    # For m > 0 (cos terms) and m < 0 (sin terms)
    for m in range(1, k + 1):
        cos_term = np.cos(m * phi)[np.newaxis, :]  # (1, n_phi)
        sin_term = np.sin(m * phi)[np.newaxis, :]  # (1, n_phi)
        P_m = P[:, m][:, np.newaxis]  # (n_theta, 1)
        coeff = np.sqrt(2.0) * norm_constants[m]
        
        # Y_m for m > 0: index is k + m
        Y[:, :, k + m] = coeff * P_m * cos_term
        # Y_m for m < 0: index is k - m
        Y[:, :, k - m] = coeff * P_m * sin_term
    
    return Y


def _compute_ratio_and_grad(coeffs, Y, w_theta, w_phi):
    """
    Compute L4/L2 ratio and its gradient.
    coeffs: shape (21,)
    Y: shape (n_theta, n_phi, 21)
    w_theta: shape (n_theta,)
    w_phi: scalar
    
    Returns (ratio, grad)
    """
    # f = sum_m c_m Y_m
    # Y has shape (n_theta, n_phi, 21), coeffs has shape (21,)
    f = np.tensordot(coeffs, Y, axes=([0], [2]))  # shape (n_theta, n_phi)
    
    # Weights for surface measure
    # weight_ij = w_theta_i * w_phi
    w_full = w_theta[:, np.newaxis] * w_phi  # shape (n_theta, 1), broadcast to (n_theta, n_phi)
    
    # L2 norm squared: sum w_ij * f_ij^2
    f2 = f * f
    norm2_sq = np.sum(w_full * f2)
    
    # L4 norm: (sum w_ij * f_ij^4)^(1/4)
    f4 = f2 * f2
    norm4 = np.sum(w_full * f4)
    
    if norm2_sq < 1e-30:
        return 1e10, np.zeros_like(coeffs)
    
    norm2 = np.sqrt(norm2_sq)
    norm4_val = norm4 ** 0.25
    
    ratio = norm4_val / norm2
    
    # Gradient of ratio = d/dc (norm4/norm2)
    # Let A = norm4 = (sum w f^4)^(1/4), B = norm2 = (sum w f^2)^(1/2)
    # dA/dc = (1/4) A^{-3} * sum w * 4 f^3 * df/dc = A^{-3} * sum w f^3 df/dc
    # dB/dc = (1/2) B^{-1} * sum w * 2 f * df/dc = B^{-1} * sum w f df/dc
    # d(A/B)/dc = (A^{-3} sum w f^3 df/dc * B - A * B^{-1} sum w f df/dc) / B^2
    #            = (A^{-3} B * sum w f^3 df/dc - A B^{-1} sum w f df/dc) / B^2
    #            = A^{-3} B^{-1} sum w f^3 df/dc - A B^{-3} sum w f df/dc
    
    # df/dc_j = Y_j (the j-th harmonic)
    # sum w f^3 df/dc_j = sum w f^3 Y_j
    # sum w f df/dc_j = sum w f Y_j
    
    # Compute these for all j at once
    # f3 = f^3, shape (n_theta, n_phi)
    f3 = f2 * f
    # sum_w_f3Y_j = sum over i,j of w_ij f_ij^3 Y_ij_j
    # Y has shape (n_theta, n_phi, 21)
    # We need: for each harmonic index h, sum w f^3 Y_h
    # This is: np.einsum('ij,ijh->h', w_full * f3, Y)
    sum_w_f3Y = np.einsum('ij,ijh->h', w_full * f3, Y)
    sum_w_fY = np.einsum('ij,ijh->h', w_full * f, Y)
    
    A = norm4_val
    B = norm2
    
    grad = A**(-3) * B**(-1) * sum_w_f3Y - A * B**(-3) * sum_w_fY
    
    return ratio, grad


def rudin_problem10() -> np.ndarray:
    """
    Find real coefficients for degree-10 spherical harmonic minimizing L4/L2 ratio.
    Returns np.ndarray of shape (21,) with coefficients c_m for m = -10..10.
    """
    deadline = time.time() + 600  # 600 second limit
    
    # Build quadrature
    x, w_theta, phi, w_phi = _build_quadrature()
    Y = _spherical_harmonics(x, phi, degree=10)
    
    # Initial guess: all ones
    x0 = np.ones(21, dtype=np.float64)
    
    # First, try L-BFGS-B with analytic gradient
    best_coeffs = x0.copy()
    best_ratio = _compute_ratio_and_grad(x0, Y, w_theta, w_phi)[0]
    
    def objective_and_grad(c):
        return _compute_ratio_and_grad(c, Y, w_theta, w_phi)
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
            x0,
            jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.success or result.fun < best_ratio:
            best_coeffs = result.x
            best_ratio = result.fun
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 60:
        return best_coeffs
    
    # Try multiple random restarts to find better local minima
    np.random.seed(42)
    n_restarts = 20
    for i in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        # Random initial guess
        x_start = np.random.randn(21)
        x_start /= np.linalg.norm(x_start)
        
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                x_start,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Final polish with tighter tolerances
    if time.time() < deadline - 30:
        try:
            result = minimize(
                lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[0],
                best_coeffs,
                jac=lambda c: _compute_ratio_and_grad(c, Y, w_theta, w_phi)[1],
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if result.fun < best_ratio:
                best_ratio = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Validate
    if len(best_coeffs) != 21:
        return x0
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        return x0
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3 or max_abs > 1e5:
        return x0
    
    return best_coeffs


# EVOLVE-BLOCK-END
