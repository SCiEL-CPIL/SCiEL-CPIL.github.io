# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.special import lpmv
from scipy.optimize import minimize

_DEADLINE = time.time() + 620  # 10s safety margin before 630s kill


def _build_quadrature():
    """70-point Gauss-Legendre in cos(theta) x 130 equally spaced phi."""
    x_gl, w_gl = np.polynomial.legendre.leggauss(70)
    cos_theta = x_gl
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta ** 2))
    theta = np.arccos(cos_theta)
    phi = np.linspace(0, 2 * np.pi, 130, endpoint=False)
    # Surface element: sin(theta) dtheta dphi -> w_gl d(cos_theta) dphi
    # Total measure should be 4*pi; check: sum(w_gl)*2*pi = 2*2*pi = 4*pi. Good.
    return cos_theta, sin_theta, theta, phi, w_gl


def _spherical_harmonics_real(cos_theta, phi, m):
    """
    Real orthonormal degree-10 spherical harmonics.
    Y_0 = N_0 P_10^0(cos theta)
    Y_m = sqrt(2) N_m P_10^m(cos theta) cos(m phi) for m > 0
    Y_m = sqrt(2) N_|m| P_10^|m|(cos theta) sin(|m| phi) for m < 0
    N_m = sqrt(21/(4*pi) * (10-m)!/(10+m)!)
    """
    k = 10
    abs_m = abs(m)
    # N_m with Condon-Shortley phase included in P via lpmv
    log_N = 0.5 * (np.log(21.0 / (4.0 * np.pi)) + 
                   _log_factorial(10 - abs_m) - _log_factorial(10 + abs_m))
    N_m = np.exp(log_N)
    
    P = lpmv(abs_m, 10, cos_theta)  # includes Condon-Shortley phase (-1)^m
    
    if m == 0:
        return N_m * P
    elif m > 0:
        return np.sqrt(2.0) * N_m * P * np.cos(m * phi)
    else:
        return np.sqrt(2.0) * N_m * P * np.sin(abs_m * phi)


def _log_factorial(n):
    """Log factorial using scipy or manual."""
    from scipy.special import gammaln
    return gammaln(n + 1)


def _compute_ratio(coeffs, cos_theta, phi, w_gl):
    """Compute ||f||_4 / ||f||_2 using quadrature."""
    # Evaluate f at all quadrature points
    # cos_theta shape (70,), phi shape (130,)
    # Build meshgrid
    ct, ph = np.meshgrid(cos_theta, phi, indexing='ij')
    # ct shape (70, 130), ph shape (70, 130)
    
    f = np.zeros_like(ct)
    for i, m in enumerate(range(-10, 11)):
        c = coeffs[i]
        if c != 0.0:
            f += c * _spherical_harmonics_real(ct, ph, m)
    
    # Weights: w_gl[i] * dphi, where dphi = 2*pi/130
    dphi = 2.0 * np.pi / 130
    weights = np.outer(w_gl, np.ones(130)) * dphi  # shape (70, 130)
    
    f2 = f ** 2
    f4 = f2 ** 2
    
    norm2_sq = np.sum(weights * f2)  # integral of f^2
    norm4_4 = np.sum(weights * f4)   # integral of f^4
    
    if norm2_sq <= 0:
        return np.inf
    
    norm2 = np.sqrt(norm2_sq)
    norm4 = np.sqrt(np.sqrt(norm4_4))  # (f^4 integral)^(1/4)
    
    return norm4 / norm2


def rudin_problem10() -> np.ndarray:
    """
    Minimize ||f||_4 / ||f||_2 for degree-10 real spherical harmonics.
    Returns 21 coefficients c_m for m = -10..10.
    """
    global _DEADLINE
    
    # Build quadrature
    cos_theta, sin_theta, theta, phi, w_gl = _build_quadrature()
    
    # Initial guess: all ones
    coeffs0 = np.ones(21, dtype=np.float64)
    
    # Compute baseline ratio
    best_ratio = _compute_ratio(coeffs0, cos_theta, phi, w_gl)
    best_coeffs = coeffs0.copy()
    
    def objective(c):
        return _compute_ratio(c, cos_theta, phi, w_gl)
    
    # Try multiple starting points and optimizers
    starts = [
        np.ones(21, dtype=np.float64),
        np.random.default_rng(42).standard_normal(21) * 0.5,
        np.random.default_rng(123).standard_normal(21) * 0.5,
        np.array([1.0 if m % 2 == 0 else -1.0 for m in range(-10, 11)], dtype=np.float64),
    ]
    
    for start in starts:
        if time.time() > _DEADLINE - 30:
            break
        
        try:
            res = minimize(
                objective,
                start,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10},
            )
            if res.success and res.fun < best_ratio:
                best_ratio = res.fun
                best_coeffs = res.x.copy()
        except Exception:
            pass
    
    # Additional refinement with different initializations
    if time.time() < _DEADLINE - 60:
        for seed in [7, 99, 256]:
            if time.time() > _DEADLINE - 30:
                break
            start = np.random.default_rng(seed).standard_normal(21) * 0.3
            try:
                res = minimize(
                    objective,
                    start,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12},
                )
                if res.fun < best_ratio:
                    best_ratio = res.fun
                    best_coeffs = res.x.copy()
            except Exception:
                pass
    
    # Ensure validity
    result = best_coeffs.astype(np.float64)
    if result.shape != (21,):
        result = np.ones(21, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        result = np.ones(21, dtype=np.float64)
    
    max_abs = np.max(np.abs(result))
    if max_abs < 1e-3:
        result = result / max_abs * 0.01 if max_abs > 0 else np.ones(21, dtype=np.float64)
    elif max_abs > 1e5:
        result = result / max_abs * 1e4
    
    return result


# EVOLVE-BLOCK-END
