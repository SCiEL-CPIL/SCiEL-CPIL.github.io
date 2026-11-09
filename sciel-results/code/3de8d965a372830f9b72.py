# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import lpmv

DEADLINE = 630.0
SAFETY_MARGIN = 30.0  # seconds to reserve before the hard kill

def _time_left():
    """Return seconds remaining before the safety deadline."""
    return DEADLINE - SAFETY_MARGIN - (time.time() - _START_TIME)

_START_TIME = time.time()

def _build_quadrature(n_theta=70, n_phi=130):
    """Build 70-point Gauss-Legendre in cos(theta) x 130 equally-spaced phi.
    Returns theta, phi, w (weights for integration over unit sphere, total 4*pi)."""
    # Gauss-Legendre nodes and weights on [-1, 1]
    x, wx = np.polynomial.legendre.leggauss(n_theta)
    # x = cos(theta), so d(cos theta) = -sin(theta) d(theta)
    # Surface element: sin(theta) dtheta dphi = d(cos theta) dphi (up to sign, use abs)
    # Actually: dOmega = sin(theta) dtheta dphi, and d(cos theta) = -sin(theta) dtheta
    # So integral over sphere = integral_{-1}^{1} integral_0^{2pi} f(cos theta, phi) dphi d(cos theta)
    # With Gauss-Legendre: sum_i sum_j w_i f(x_i, phi_j) where w_i are GL weights
    
    # Weights: the GL weights wx already account for the d(cos theta) integration
    # We need to integrate over phi from 0 to 2pi with equal spacing
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    dphi = 2 * np.pi / n_phi
    
    # Full 2D weights: w[i,j] = wx[i] * dphi
    # Total mass should be 4*pi: sum(wx) = 2, so sum(w) = 2 * 2*pi = 4*pi. Correct.
    
    # Create meshgrid
    cos_theta, phi_grid = np.meshgrid(x, phi, indexing='ij')
    sin_theta = np.sqrt(np.maximum(0, 1 - x**2))
    theta = np.arccos(x)
    
    # Weights for each quadrature point
    w = np.outer(wx, np.ones(n_phi)) * dphi
    
    return cos_theta, phi_grid, sin_theta, w

def _spherical_harmonics_real(cos_theta, phi, sin_theta, degree=10):
    """Compute real orthonormal spherical harmonics Y_m for m=-degree..degree.
    Y_0 = N_0 P^0
    Y_m = sqrt(2) N_m P^m cos(m*phi) for m > 0
    Y_m = sqrt(2) N_|m| P^|m| sin(|m|*phi) for m < 0
    where N_m = sqrt((2*degree+1)/(4*pi) * (degree-m)!/(degree+m)!)
    """
    k = degree
    n_m = 2 * k + 1
    Y = np.zeros((n_m,) + cos_theta.shape, dtype=np.float64)
    
    for m in range(-k, k + 1):
        abs_m = abs(m)
        # Normalization: N_m = sqrt((2k+1)/(4*pi) * (k-abs_m)!/(k+abs_m)!)
        # Using log-gamma for stability
        from math import lgamma
        log_N = 0.5 * (np.log((2*k+1)/(4*np.pi)) + lgamma(k - abs_m + 1) - lgamma(k + abs_m + 1))
        N_m = np.exp(log_N)
        
        # Associated Legendre function with Condon-Shortley phase
        # lpmv(m, n, x) gives P_n^m(x) with Condon-Shortley phase (-1)^m
        P_m = lpmv(abs_m, k, cos_theta)
        
        idx = m + k
        if m == 0:
            Y[idx] = N_m * P_m
        elif m > 0:
            Y[idx] = np.sqrt(2) * N_m * P_m * np.cos(m * phi)
        else:
            Y[idx] = np.sqrt(2) * N_m * P_m * np.sin(abs_m * phi)
    
    return Y

def _compute_f(coeffs, cos_theta, phi, sin_theta, Y):
    """Compute f = sum_m c_m Y_m at quadrature points."""
    # Y has shape (21, n_theta, n_phi), coeffs has shape (21,)
    f = np.tensordot(coeffs, Y, axes=([0], [0]))
    return f

def _ratio_from_coeffs(coeffs, cos_theta, phi, sin_theta, Y, w):
    """Compute R = ||f||_4 / ||f||_2."""
    f = _compute_f(coeffs, cos_theta, phi, sin_theta, Y)
    
    # L2 norm squared: integral f^2 dOmega
    l2_sq = np.sum(w * f**2)
    if l2_sq <= 0:
        return np.inf
    
    # L4 norm to the 4th power: integral f^4 dOmega
    l4_4 = np.sum(w * f**4)
    if l4_4 <= 0:
        return 0.0
    
    l2 = np.sqrt(l2_sq)
    l4 = np.sqrt(np.sqrt(l4_4))  # (f^4 integral)^(1/4) = (l4_4)^(1/4)
    
    return l4 / l2

def _objective_and_gradient(coeffs, cos_theta, phi, sin_theta, Y, w):
    """Compute R = ||f||_4/||f||_2 and its gradient."""
    f = _compute_f(coeffs, cos_theta, phi, sin_theta, Y)
    
    l2_sq = np.sum(w * f**2)
    if l2_sq <= 0:
        return np.inf, np.zeros_like(coeffs)
    
    l4_4 = np.sum(w * f**4)
    
    l2 = np.sqrt(l2_sq)
    l4 = l4_4**0.25
    
    R = l4 / l2
    
    # Gradient of R = l4 / l2
    # dR = (1/l2) * dl4 - (l4/l2^2) * dl2
    # dl2 = (1/(2*l2)) * d(l2_sq) = (1/(2*l2)) * sum(w * 2f * df) = sum(w * f * df) / l2
    # dl4 = (1/4) * l4_4^(-3/4) * d(l4_4) = (1/4) * l4_4^(-3/4) * sum(w * 4f^3 * df) = l4_4^(-3/4) * sum(w * f^3 * df)
    
    # df/dc_m = Y_m
    # dl2_sq/dc_m = 2 * sum(w * f * Y_m)
    # dl4_4/dc_m = 4 * sum(w * f^3 * Y_m)
    
    # grad R = (1/l2) * (1/4) * l4_4^(-3/4) * 4 * sum(w * f^3 * Y_m) - (l4/l2^2) * (1/(2*l2)) * 2 * sum(w * f * Y_m)
    # grad R = (1/l2) * l4_4^(-3/4) * sum(w * f^3 * Y_m) - (l4/l2^2) * (1/l2) * sum(w * f * Y_m)
    # grad R = l4_4^(-3/4) / l2 * sum(w * f^3 * Y_m) - l4 / l2^3 * sum(w * f * Y_m)
    
    # Note: l4_4^(-3/4) = 1/l4^3
    # And l4 = l4_4^(1/4), so l4_4^(-3/4) = l4^(-3)
    
    # grad R = (1/(l2 * l4^3)) * sum(w * f^3 * Y_m) - (l4 / l2^3) * sum(w * f * Y_m)
    #        = (1/(l2 * l4^3)) * [sum(w * f^3 * Y_m) - (l4^4 / l2^2) * sum(w * f * Y_m)]
    #        = (1/(l2 * l4^3)) * [sum(w * f^3 * Y_m) - (l4_4 / l2_sq) * sum(w * f * Y_m)]
    
    # Let's compute the vectors
    # v1[m] = sum(w * f^3 * Y_m)
    # v2[m] = sum(w * f * Y_m)
    
    w_f3 = w * f**3
    w_f = w * f
    
    # Y shape: (21, n_theta, n_phi)
    # w_f3 shape: (n_theta, n_phi)
    # v1[m] = sum over (i,j) of w_f3[i,j] * Y[m,i,j]
    
    v1 = np.tensordot(Y, w_f3, axes=([1, 2], [0, 1]))
    v2 = np.tensordot(Y, w_f, axes=([1, 2], [0, 1]))
    
    factor1 = 1.0 / (l2 * l4**3)
    factor2 = l4 / l2**3
    
    grad = factor1 * v1 - factor2 * v2
    
    return R, grad

def _optimize_from(start_coeffs, cos_theta, phi, sin_theta, Y, w, max_iter=2000, tol=1e-12):
    """Run L-BFGS-B optimization from a starting point."""
    best_R = _ratio_from_coeffs(start_coeffs, cos_theta, phi, sin_theta, Y, w)
    best_coeffs = start_coeffs.copy()
    
    def obj_and_grad(c):
        if _time_left() < 5.0:
            raise RuntimeError("Time limit")
        R, g = _objective_and_gradient(c, cos_theta, phi, sin_theta, Y, w)
        return R, g
    
    try:
        result = minimize(
            lambda c: _objective_and_gradient(c, cos_theta, phi, sin_theta, Y, w)[0],
            start_coeffs,
            jac=lambda c: _objective_and_gradient(c, cos_theta, phi, sin_theta, Y, w)[1],
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxfun': 5000}
        )
        
        if result.success or result.fun < best_R:
            R_res = _ratio_from_coeffs(result.x, cos_theta, phi, sin_theta, Y, w)
            if R_res < best_R:
                best_R = R_res
                best_coeffs = result.x.copy()
    except Exception:
        pass
    
    return best_coeffs, best_R

def _make_seeds(n_seeds=30):
    """Generate diverse starting points."""
    seeds = []
    k = 10
    n = 2 * k + 1
    
    # All ones
    seeds.append(np.ones(n))
    
    # All zeros except one
    for m in range(n):
        c = np.zeros(n)
        c[m] = 1.0
        seeds.append(c)
    
    # Random with various scales
    rng = np.random.default_rng(42)
    for i in range(15):
        c = rng.standard_normal(n)
        c = c / np.max(np.abs(c))
        seeds.append(c)
    
    # Symmetric patterns
    for i in range(5):
        c = np.zeros(n)
        c[0] = 1.0
        c[-1] = 1.0
        c[10] = rng.uniform(-2, 2)
        seeds.append(c)
    
    return seeds

def rudin_problem10() -> np.ndarray:
    global _START_TIME
    _START_TIME = time.time()
    
    # Build quadrature
    cos_theta, phi, sin_theta, w = _build_quadrature(70, 130)
    
    # Precompute spherical harmonics
    Y = _spherical_harmonics_real(cos_theta, phi, sin_theta, degree=10)
    
    # Generate seeds
    seeds = _make_seeds(30)
    
    best_coeffs = np.ones(21)
    best_R = _ratio_from_coeffs(best_coeffs, cos_theta, phi, sin_theta, Y, w)
    
    # Multi-start optimization
    for i, seed in enumerate(seeds):
        if _time_left() < 30.0:
            break
        
        coeffs, R = _optimize_from(seed, cos_theta, phi, sin_theta, Y, w, max_iter=1500, tol=1e-10)
        
        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
    
    # Final polish with tighter tolerance
    if _time_left() > 60.0:
        coeffs, R = _optimize_from(best_coeffs, cos_theta, phi, sin_theta, Y, w, max_iter=3000, tol=1e-14)
        if R < best_R:
            best_R = R
            best_coeffs = coeffs.copy()
    
    # Ensure validity
    best_coeffs = np.asarray(best_coeffs, dtype=np.float64)
    if best_coeffs.shape != (21,):
        best_coeffs = np.ones(21)
    if np.any(np.isnan(best_coeffs)) or np.any(np.isinf(best_coeffs)):
        best_coeffs = np.ones(21)
    
    max_abs = np.max(np.abs(best_coeffs))
    if max_abs < 1e-3:
        best_coeffs = best_coeffs * (1.0 / max_abs)
    elif max_abs > 1e5:
        best_coeffs = best_coeffs * (1e5 / max_abs)
    
    return best_coeffs

# EVOLVE-BLOCK-END
