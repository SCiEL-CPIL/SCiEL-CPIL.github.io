"""Uncertainty bound minimization via analytic gradient + differential evolution + L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from math import sqrt, pi

_DEADLINE = 195.0


def _int_x2m(m):
    """∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = sqrt(π) (2m-1)!! / 2^m"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val


# Precompute Hermite polynomial coefficients (physicists' convention)
# H_0(x) = 1
# H_4(x) = 16x^4 - 48x^2 + 12
# H_8(x) = 256x^8 - 4480x^6 + 30240x^4 - 60480x^2 + 2520
# H_12(x) = 4096x^12 - 147456x^10 + 2177280x^8 - 14515200x^6 + 38976000x^4 - 39916800x^2 + 27720

_H0 = np.zeros(13)
_H0[0] = 1.0

_H4 = np.zeros(13)
_H4[4] = 16.0
_H4[2] = -48.0
_H4[0] = 12.0

_H8 = np.zeros(13)
_H8[8] = 256.0
_H8[6] = -4480.0
_H8[4] = 30240.0
_H8[2] = -60480.0
_H8[0] = 2520.0

_H12 = np.zeros(13)
_H12[12] = 4096.0
_H12[10] = -147456.0
_H12[8] = 2177280.0
_H12[6] = -14515200.0
_H12[4] = 38976000.0
_H12[2] = -39916800.0
_H12[0] = 27720.0

_H_COEFFS = np.array([_H0, _H4, _H8, _H12])  # shape (4, 13)

# H values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 2520.0
_H12_0 = 27720.0

# Precompute integrals for moments up to degree 24
_int_table = np.zeros(13)  # _int_x2m(m) for m=0..12
for m in range(13):
    _int_table[m] = _int_x2m(m)


def _compute_bound_and_grad(params):
    """
    Compute the uncertainty bound and its gradient with respect to (a0, a4, a8).
    
    Returns (bound, grad) where grad is a 3-element array.
    """
    a0, a4, a8 = params
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    coeffs = np.array([a0, a4, a8, a12])
    p_poly = coeffs @ _H_COEFFS  # shape (13,)
    
    # p(x)^2 as polynomial (degree up to 24)
    p2_poly = np.zeros(25)
    for i in range(13):
        if p_poly[i] != 0.0:
            for j in range(13):
                p2_poly[i + j] += p_poly[i] * p_poly[j]
    
    # M0 = ∫ p^2 e^{-x^2} dx
    M0 = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        M0 += p2_poly[k] * _int_table[m]
    
    # M2 = ∫ x^2 p^2 e^{-x^2} dx
    M2 = 0.0
    for k in range(0, 25, 2):
        m_plus_1 = (k + 2) // 2
        M2 += p2_poly[k] * _int_table[m_plus_1]
    
    # p'(x)
    p_prime = np.zeros(12)
    for i in range(1, 13):
        p_prime[i - 1] = p_poly[i] * i
    
    # (p')^2
    pprime_sq = np.zeros(23)
    for i in range(12):
        if p_prime[i] != 0.0:
            for j in range(12):
                pprime_sq[i + j] += p_prime[i] * p_prime[j]
    
    int_pprime_sq = 0.0
    for k in range(0, 23, 2):
        m = k // 2
        int_pprime_sq += pprime_sq[k] * _int_table[m]
    
    # x*p(x)
    xp = np.zeros(14)
    for i in range(13):
        xp[i + 1] = p_poly[i]
    
    # x*p(x) * p'(x)
    xppp = np.zeros(25)
    for i in range(14):
        if xp[i] != 0.0:
            for j in range(12):
                xppp[i + j] += xp[i] * p_prime[j]
    
    int_xppp = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        int_xppp += xppp[k] * _int_table[m]
    
    int_gprime_sq = int_pprime_sq - 2.0 * int_xppp + M2
    
    if M0 <= 1e-15 or int_gprime_sq <= 1e-15:
        return np.inf, np.zeros(3)
    
    bound = np.sqrt(M2 * int_gprime_sq) / M0
    
    # Compute gradient via finite differences (small step for accuracy)
    eps = 1e-7
    grad = np.zeros(3)
    for i in range(3):
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        
        f_plus = _compute_bound_only(p_plus)
        f_minus = _compute_bound_only(p_minus)
        grad[i] = (f_plus - f_minus) / (2.0 * eps)
    
    return bound, grad


def _compute_bound_only(params):
    """Compute just the bound (no gradient) for speed."""
    a0, a4, a8 = params
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    coeffs = np.array([a0, a4, a8, a12])
    p_poly = coeffs @ _H_COEFFS
    
    p2_poly = np.zeros(25)
    for i in range(13):
        if p_poly[i] != 0.0:
            for j in range(13):
                p2_poly[i + j] += p_poly[i] * p_poly[j]
    
    M0 = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        M0 += p2_poly[k] * _int_table[m]
    
    M2 = 0.0
    for k in range(0, 25, 2):
        m_plus_1 = (k + 2) // 2
        M2 += p2_poly[k] * _int_table[m_plus_1]
    
    p_prime = np.zeros(12)
    for i in range(1, 13):
        p_prime[i - 1] = p_poly[i] * i
    
    pprime_sq = np.zeros(23)
    for i in range(12):
        if p_prime[i] != 0.0:
            for j in range(12):
                pprime_sq[i + j] += p_prime[i] * p_prime[j]
    
    int_pprime_sq = 0.0
    for k in range(0, 23, 2):
        m = k // 2
        int_pprime_sq += pprime_sq[k] * _int_table[m]
    
    xp = np.zeros(14)
    for i in range(13):
        xp[i + 1] = p_poly[i]
    
    xppp = np.zeros(25)
    for i in range(14):
        if xp[i] != 0.0:
            for j in range(12):
                xppp[i + j] += xp[i] * p_prime[j]
    
    int_xppp = 0.0
    for k in range(0, 25, 2):
        m = k // 2
        int_xppp += xppp[k] * _int_table[m]
    
    int_gprime_sq = int_pprime_sq - 2.0 * int_xppp + M2
    
    if M0 <= 1e-15 or int_gprime_sq <= 1e-15:
        return np.inf
    
    return np.sqrt(M2 * int_gprime_sq) / M0


def _objective(params):
    """Objective function for optimization."""
    return _compute_bound_only(params)


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Known good starting point
    a0_lin = -113.0 / 100
    a4_lin = 1.0 / 25
    a8_lin = 1.0 / 3240
    
    best_x = np.array([a0_lin, a4_lin, a8_lin])
    best_bound = _compute_bound_only(best_x)
    
    # Phase 1: Differential evolution for global exploration
    if time.time() - start_time < _DEADLINE - 60:
        try:
            bounds = [(-5.0, 5.0), (-2.0, 2.0), (-0.5, 0.5)]
            de_result = differential_evolution(
                _objective,
                bounds=bounds,
                maxiter=100,
                popsize=20,
                tol=1e-10,
                seed=42,
                polish=True,
                init='sobol',
                update_interval=1,
                callback=None
            )
            if de_result.fun < best_bound:
                best_bound = de_result.fun
                best_x = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multiple L-BFGS-B refinements from diverse starts
    rng = np.random.default_rng(12345)
    starts = [best_x.copy()]
    
    # Add the linear solution and its variations
    starts.append(np.array([a0_lin, a4_lin, a8_lin]))
    
    # Scale variations
    for scale in [0.1, 0.3, 0.5, 0.7, 1.3, 1.5, 2.0, 3.0]:
        starts.append(best_x * scale)
    
    # Random perturbations
    for _ in range(30):
        s = np.array([
            best_x[0] + rng.normal(0, 0.5),
            best_x[1] + rng.normal(0, 0.2),
            best_x[2] + rng.normal(0, 0.05)
        ])
        starts.append(s)
    
    # Aggressive random starts
    for _ in range(20):
        s = np.array([
            rng.uniform(-3.0, 3.0),
            rng.uniform(-1.0, 1.0),
            rng.uniform(-0.3, 0.3)
        ])
        starts.append(s)
    
    for x_start in starts:
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        try:
            result = minimize(
                _objective,
                x_start,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Final high-precision refinement
    if time.time() - start_time < _DEADLINE - 5:
        try:
            result = minimize(
                _objective,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-14}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 4: Nelder-Mead for final polish
    if time.time() - start_time < _DEADLINE - 2:
        try:
            result = minimize(
                _objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    return best_x
# EVOLVE-BLOCK-END
