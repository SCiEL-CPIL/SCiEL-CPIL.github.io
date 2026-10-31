"""Uncertainty bound minimization via stationary-point root-finding + multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root, minimize
from math import sqrt, pi

_DEADLINE = 190.0


def _hermite_coeffs(n):
    """Return coefficients of the physicists' Hermite polynomial H_n(x) as dict power -> coeff."""
    if n == 0:
        return {0: 1.0}
    if n == 1:
        return {1: 2.0}
    h_prev = {0: 1.0}
    h_curr = {1: 2.0}
    for m in range(2, n + 1):
        h_new = {}
        for p, c in h_curr.items():
            new_p = p + 1
            h_new[new_p] = h_new.get(new_p, 0.0) + 2.0 * c
        for p, c in h_prev.items():
            h_new[p] = h_new.get(p, 0.0) - 2.0 * (m - 1) * c
        h_new = {p: c for p, c in h_new.items() if abs(c) > 1e-15}
        h_prev, h_curr = h_curr, h_new
    return h_curr


def _int_x2m(m):
    """∫_{-∞}^{∞} x^{2m} e^{-x^2} dx = sqrt(π) (2m-1)!! / 2^m"""
    if m == 0:
        return sqrt(pi)
    val = sqrt(pi)
    for j in range(1, m + 1):
        val *= (2 * j - 1) / 2.0
    return val


# Precompute Hermite coefficient dicts for degrees 0, 4, 8, 12
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Precompute p_poly basis: for each degree, store the contribution to p_poly
# p_poly has length 13 (powers 0..12)
_DEG_MAX = 12
_POLY_LEN = _DEG_MAX + 1

# For each of the 4 Hermite polynomials, precompute the coefficient vector
_H_COEFFS = []
for _hc in [_H0, _H4, _H8, _H12]:
    vec = np.zeros(_POLY_LEN)
    for p, c in _hc.items():
        if p < _POLY_LEN:
            vec[p] = c
    _H_COEFFS.append(vec)
_H_COEFFS = np.array(_H_COEFFS)  # shape (4, 13)

# H values at zero for the constraint
_H0_0 = 1.0
_H4_0 = 12.0   # (-1)^2 * 4!/2!
_H8_0 = 2520.0  # (-1)^4 * 8!/4!
_H12_0 = 27720.0  # (-1)^6 * 12!/6!


def _compute_moments(a0, a4, a8):
    """Compute the uncertainty bound given coefficients a0, a4, a8."""
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    
    # Build p_poly = a0*H0 + a4*H4 + a8*H8 + a12*H12
    coeffs = np.array([a0, a4, a8, a12])
    p_poly = coeffs @ _H_COEFFS  # shape (13,)
    
    # p(x)^2 as polynomial
    p2_poly = np.zeros(2 * _DEG_MAX + 1)
    for i in range(_POLY_LEN):
        if p_poly[i] != 0.0:
            for j in range(_POLY_LEN):
                p2_poly[i + j] += p_poly[i] * p_poly[j]
    
    # M0 = ∫ p^2 e^{-x^2} dx
    M0 = 0.0
    for k in range(0, 2 * _DEG_MAX + 1, 2):
        m = k // 2
        M0 += p2_poly[k] * _int_x2m(m)
    
    # M2 = ∫ x^2 p^2 e^{-x^2} dx
    M2 = 0.0
    for k in range(0, 2 * _DEG_MAX + 1, 2):
        m_plus_1 = (k + 2) // 2
        M2 += p2_poly[k] * _int_x2m(m_plus_1)
    
    # g'(x) = (p' - x p) e^{-x^2/2}
    # p'(x)
    p_prime = np.zeros(_POLY_LEN - 1)
    for i in range(1, _POLY_LEN):
        p_prime[i - 1] = p_poly[i] * i
    
    # (p')^2
    pprime_sq = np.zeros(2 * (_POLY_LEN - 1) - 1)
    for i in range(_POLY_LEN - 1):
        if p_prime[i] != 0.0:
            for j in range(_POLY_LEN - 1):
                pprime_sq[i + j] += p_prime[i] * p_prime[j]
    
    int_pprime_sq = 0.0
    for k in range(0, len(pprime_sq), 2):
        m = k // 2
        int_pprime_sq += pprime_sq[k] * _int_x2m(m)
    
    # x*p(x)
    xp = np.zeros(_POLY_LEN + 1)
    for i in range(_POLY_LEN):
        xp[i + 1] = p_poly[i]
    
    # x*p(x) * p'(x)
    xppp = np.zeros(_POLY_LEN + (_POLY_LEN - 1))
    for i in range(_POLY_LEN + 1):
        if xp[i] != 0.0:
            for j in range(_POLY_LEN - 1):
                xppp[i + j] += xp[i] * p_prime[j]
    
    int_xppp = 0.0
    for k in range(0, len(xppp), 2):
        m = k // 2
        int_xppp += xppp[k] * _int_x2m(m)
    
    int_gprime_sq = int_pprime_sq - 2.0 * int_xppp + M2
    
    if M0 <= 1e-15 or int_gprime_sq <= 1e-15:
        return np.inf
    
    bound = np.sqrt(M2 * int_gprime_sq) / M0
    return bound


def _objective_and_grad(params):
    """
    params = [s0, m0, s4, m4, s8, m8] where a_i = s_i * exp(m_i)
    Returns (objective_value, gradient_wrt_params).
    """
    a0 = params[0] * np.exp(params[1])
    a4 = params[2] * np.exp(params[3])
    a8 = params[4] * np.exp(params[5])
    
    bound = _compute_moments(a0, a4, a8)
    
    # Analytic gradient via chain rule: d(bound)/d(params)
    # d(a_i)/d(s_i) = exp(m_i), d(a_i)/d(m_i) = s_i * exp(m_i) = a_i
    # So we need d(bound)/d(a0), d(bound)/d(a4), d(bound)/d(a8)
    # Use central finite differences for d(bound)/d(a_i) since analytic is complex
    h = 1e-8
    da0_plus = _compute_moments(a0 + h, a4, a8)
    da0_minus = _compute_moments(a0 - h, a4, a8)
    da4_plus = _compute_moments(a0, a4 + h, a8)
    da4_minus = _compute_moments(a0, a4 - h, a8)
    da8_plus = _compute_moments(a0, a4, a8 + h)
    da8_minus = _compute_moments(a0, a4, a8 - h)
    
    dbound_da0 = (da0_plus - da0_minus) / (2 * h)
    dbound_da4 = (da4_plus - da4_minus) / (2 * h)
    dbound_da8 = (da8_plus - da8_minus) / (2 * h)
    
    grad = np.zeros(6)
    grad[0] = dbound_da0 * np.exp(params[1])
    grad[1] = dbound_da0 * a0
    grad[2] = dbound_da4 * np.exp(params[3])
    grad[3] = dbound_da4 * a4
    grad[4] = dbound_da8 * np.exp(params[5])
    grad[5] = dbound_da8 * a8
    
    return bound, grad


def _objective(params):
    """Objective for scipy minimize (returns scalar only)."""
    a0 = params[0] * np.exp(params[1])
    a4 = params[2] * np.exp(params[3])
    a8 = params[4] * np.exp(params[5])
    return _compute_moments(a0, a4, a8)


def _objective_grad(params):
    """Objective with gradient for L-BFGS-B."""
    val, grad = _objective_and_grad(params)
    return val, grad


def _objective_linear(params):
    """Linear-space objective for Nelder-Mead."""
    return _compute_moments(params[0], params[1], params[2])


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Known good solution in linear space
    a0_lin = -113.0 / 100
    a4_lin = 1.0 / 25
    a8_lin = 1.0 / 3240
    
    best_x = np.array([a0_lin, a4_lin, a8_lin])
    best_bound = _compute_moments(a0_lin, a4_lin, a8_lin)
    
    # Generate diverse starting points in linear space
    rng = np.random.default_rng(42)
    starts = [best_x.copy()]
    
    # Sign variations
    for s0 in [-1.0, 1.0]:
        for s4 in [-1.0, 1.0]:
            for s8 in [-1.0, 1.0]:
                pt = np.array([s0 * abs(a0_lin), s4 * abs(a4_lin), s8 * abs(a8_lin)])
                starts.append(pt)
    
    # Magnitude scaling
    for scale in [0.3, 0.5, 0.7, 1.3, 1.7, 2.5, 4.0]:
        starts.append(best_x * scale)
        starts.append(np.array([-best_x[0] * scale, best_x[1] * scale, best_x[2] * scale]))
    
    # Random perturbations in log-magnitude space
    for _ in range(30):
        log_pt = np.log(np.abs(best_x) + 1e-12) + rng.normal(0, 1.0, size=3)
        signs = np.sign(best_x)
        pt = signs * np.exp(log_pt)
        starts.append(pt)
    
    # Even more diverse random starts
    for _ in range(20):
        pt = rng.uniform(-5, 5, size=3)
        starts.append(pt)
    
    # Phase 1: Root-finding on stationary equations from multiple starts
    for x_start in starts:
        if time.time() - start_time > _DEADLINE - 40:
            break
        try:
            sol = root(
                _stationary_equations,
                x_start,
                method='hybr',
                options={'xtol': 1e-12, 'maxfev': 500}
            )
            if sol.success or np.all(np.abs(sol.fun) < 1e-6):
                a0_v, a4_v, a8_v = sol.x
                b = _compute_moments(a0_v, a4_v, a8_v)
                if np.isfinite(b) and b < best_bound:
                    best_bound = b
                    best_x = np.array([a0_v, a4_v, a8_v])
        except Exception:
            pass
    
    # Phase 2: Refine best with Nelder-Mead
    if time.time() - start_time < _DEADLINE - 20:
        try:
            result = minimize(
                _objective_linear,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16, 'adaptive': True}
            )
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_x = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Another round of root-finding from refined point and perturbations
    if time.time() - start_time < _DEADLINE - 10:
        refine_starts = [best_x.copy()]
        for scale in [0.8, 1.2]:
            refine_starts.append(best_x * scale)
        for _ in range(10):
            refine_starts.append(best_x + rng.normal(0, 0.01 * np.abs(best_x) + 1e-4, size=3))
        
        for x_start in refine_starts:
            if time.time() - start_time > _DEADLINE - 5:
                break
            try:
                sol = root(
                    _stationary_equations,
                    x_start,
                    method='hybr',
                    options={'xtol': 1e-14, 'maxfev': 300}
                )
                if sol.success or np.all(np.abs(sol.fun) < 1e-8):
                    a0_v, a4_v, a8_v = sol.x
                    b = _compute_moments(a0_v, a4_v, a8_v)
                    if np.isfinite(b) and b < best_bound:
                        best_bound = b
                        best_x = np.array([a0_v, a4_v, a8_v])
            except Exception:
                pass
    
    return best_x
# EVOLVE-BLOCK-END
