"""Uncertainty bound minimization via constrained quadratic eigenvalue approach + local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from math import sqrt, pi
from scipy.optimize import minimize, root

_DEADLINE = 200.0


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

_DEG_MAX = 12
_POLY_LEN = _DEG_MAX + 1

# Precompute coefficient vectors for each Hermite polynomial
_H_COEFFS = []
for _hc in [_H0, _H4, _H8, _H12]:
    vec = np.zeros(_POLY_LEN)
    for p, c in _hc.items():
        if p < _POLY_LEN:
            vec[p] = c
    _H_COEFFS.append(vec)
_H_COEFFS = np.array(_H_COEFFS)  # shape (4, 13)

# H values at zero
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 2520.0
_H12_0 = 27720.0


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


def _stationary_equations(x):
    """
    Stationary equations for the optimization problem.
    x = [a0, a4, a8]
    We minimize f(a0,a4,a8) = sqrt(M2 * G) / M0 subject to g(0)=0 (already enforced via a12).
    
    The gradient of the objective with respect to a0, a4, a8 should be zero at optimum.
    We use finite differences to compute the gradient components.
    """
    a0, a4, a8 = x
    h = 1e-7
    
    f0 = _compute_moments(a0, a4, a8)
    
    # Central differences for gradient
    df_da0 = (_compute_moments(a0 + h, a4, a8) - _compute_moments(a0 - h, a4, a8)) / (2*h)
    df_da4 = (_compute_moments(a0, a4 + h, a8) - _compute_moments(a0, a4 - h, a8)) / (2*h)
    df_da8 = (_compute_moments(a0, a4, a8 + h) - _compute_moments(a0, a4, a8 - h)) / (2*h)
    
    return [df_da0, df_da4, df_da8]


def _eigenvalue_approach():
    """
    Formulate as a generalized eigenvalue problem.
    
    The uncertainty bound is sqrt(M2 * G) / M0 where:
    - M0 = ∫ p^2 e^{-x^2} dx
    - M2 = ∫ x^2 p^2 e^{-x^2} dx  
    - G = ∫ (g')^2 dx = ∫ (p' - xp)^2 e^{-x^2} dx
    
    With the constraint g(0) = 0, we have a12 = -(a0 + 12*a4 + 2520*a8)/27720.
    
    All quantities are quadratic in (a0, a4, a8), so we can write:
    M0 = v^T A0 v
    M2 = v^T A2 v
    G  = v^T AG v
    where v = [a0, a4, a8].
    
    The bound is sqrt((v^T A2 v)(v^T AG v)) / (v^T A0 v)
    
    This is a ratio of quadratic forms. We can minimize this by finding the
    eigenvector that minimizes the Rayleigh quotient-like quantity.
    
    Let's construct the matrices A0, A2, AG explicitly.
    """
    # We need to express M0, M2, G as quadratic forms in v = [a0, a4, a8].
    # p_poly = a0*H0_vec + a4*H4_vec + a8*H8_vec + a12*H12_vec
    # where a12 = -(a0*H0_0 + a4*H4_0 + a8*H8_0)/H12_0
    
    # Let's define the effective coefficient vector for each basis element.
    # p_poly = sum_i v_i * e_i where e_i is the contribution to p_poly from a_i.
    
    # e_0 = H0_vec - (H0_0/H12_0)*H12_vec
    # e_4 = H4_vec - (H4_0/H12_0)*H12_vec  
    # e_8 = H8_vec - (H8_0/H12_0)*H12_vec
    
    H0_vec = _H_COEFFS[0]
    H4_vec = _H_COEFFS[1]
    H8_vec = _H_COEFFS[2]
    H12_vec = _H_COEFFS[3]
    
    e0 = H0_vec - (_H0_0 / _H12_0) * H12_vec
    e4 = H4_vec - (_H4_0 / _H12_0) * H12_vec
    e8 = H8_vec - (_H8_0 / _H12_0) * H12_vec
    
    # p_poly = a0*e0 + a4*e4 + a8*e8
    
    # Now compute M0 = ∫ p^2 e^{-x^2} dx
    # p^2 = (a0*e0 + a4*e4 + a8*e8)^2
    # = a0^2 * e0^2 + a4^2 * e4^2 + a8^2 * e8^2 + 2*a0*a4*e0*e4 + 2*a0*a8*e0*e8 + 2*a4*a8*e4*e8
    
    # For a polynomial q(x) = sum_k q_k x^k, ∫ q(x) e^{-x^2} dx = sum over even k of q_k * int_x2m(k/2)
    
    def _poly_sq_integral(p1, p2):
        """Compute ∫ p1(x)*p2(x) e^{-x^2} dx where p1, p2 are coefficient arrays (low to high power)."""
        # Product polynomial
        prod = np.zeros(len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            if p1[i] != 0.0:
                for j in range(len(p2)):
                    if p2[j] != 0.0:
                        prod[i+j] += p1[i] * p2[j]
        result = 0.0
        for k in range(0, len(prod), 2):
            m = k // 2
            result += prod[k] * _int_x2m(m)
        return result
    
    def _poly_x2_sq_integral(p1, p2):
        """Compute ∫ x^2 * p1(x)*p2(x) e^{-x^2} dx."""
        prod = np.zeros(len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            if p1[i] != 0.0:
                for j in range(len(p2)):
                    if p2[j] != 0.0:
                        prod[i+j] += p1[i] * p2[j]
        result = 0.0
        for k in range(0, len(prod), 2):
            m = (k + 2) // 2
            result += prod[k] * _int_x2m(m)
        return result
    
    # M0 matrix: A0[i][j] = ∫ e_i * e_j * e^{-x^2} dx
    A0 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei = [e0, e4, e8][i]
            ej = [e0, e4, e8][j]
            A0[i][j] = _poly_sq_integral(ei, ej)
    
    # M2 matrix: A2[i][j] = ∫ x^2 * e_i * e_j * e^{-x^2} dx
    A2 = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei = [e0, e4, e8][i]
            ej = [e0, e4, e8][j]
            A2[i][j] = _poly_x2_sq_integral(ei, ej)
    
    # G matrix: AG[i][j] = ∫ (e_i' - x*e_i) * (e_j' - x*e_j) * e^{-x^2} dx
    # e_i' - x*e_i is the derivative part. Let's compute this for each e_i.
    
    def _diff_minus_x(p):
        """Compute q(x) = p'(x) - x*p(x) given p as coefficient array."""
        n = len(p)
        q = np.zeros(n)  # q has same degree as p
        # p'(x) has degree n-2
        for k in range(1, n):
            q[k-1] += p[k] * k
        # -x*p(x) has degree n
        for k in range(n):
            q[k+1] += -p[k] if k+1 < len(q) else 0
        # Actually, x*p(x) has degree n, so q should have degree n
        q = np.zeros(n + 1)
        for k in range(1, n):
            q[k-1] += p[k] * k
        for k in range(n):
            q[k+1] += -p[k]
        return q
    
    # Recompute with correct size
    def _diff_minus_x_correct(p):
        """Compute q(x) = p'(x) - x*p(x). p has degree d, so p' has degree d-1, x*p has degree d+1."""
        n = len(p)  # p has powers 0..n-1
        # p' has powers 0..n-2
        # x*p has powers 1..n
        # So q has powers 0..n
        q = np.zeros(n + 1)
        for k in range(1, n):
            q[k-1] += p[k] * k
        for k in range(n):
            q[k+1] += -p[k]
        return q
    
    def _int_prod(q1, q2):
        """∫ q1(x) q2(x) e^{-x^2} dx"""
        prod = np.zeros(len(q1) + len(q2) - 1)
        for i in range(len(q1)):
            if q1[i] != 0.0:
                for j in range(len(q2)):
                    if q2[j] != 0.0:
                        prod[i+j] += q1[i] * q2[j]
        result = 0.0
        for k in range(0, len(prod), 2):
            m = k // 2
            result += prod[k] * _int_x2m(m)
        return result
    
    q0 = _diff_minus_x_correct(e0)
    q4 = _diff_minus_x_correct(e4)
    q8 = _diff_minus_x_correct(e8)
    q_list = [q0, q4, q8]
    
    AG = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            AG[i][j] = _int_prod(q_list[i], q_list[j])
    
    # Now we want to minimize sqrt((v^T A2 v)(v^T AG v)) / (v^T A0 v)
    # This is equivalent to minimizing (v^T A2 v)(v^T AG v) / (v^T A0 v)^2
    # Let r = (v^T A2 v)(v^T AG v) / (v^T A0 v)^2
    # We can use the substitution and find stationary points.
    
    # One approach: minimize r over the sphere v^T A0 v = 1, then the bound is sqrt(r).
    # But this is still non-convex.
    
    # Alternative: use the fact that for a fixed ratio, we can set up a system.
    # Let's just use the matrices to do a grid search + local optimization.
    
    # First, do a coarse grid search to find good starting points.
    best_v = None
    best_bound = np.inf
    
    # Grid search in a reasonable range
    for a0 in np.linspace(-2, 0, 15):
        for a4 in np.linspace(-0.5, 0.5, 15):
            for a8 in np.linspace(-0.01, 0.01, 15):
                b = _compute_moments(a0, a4, a8)
                if np.isfinite(b) and b < best_bound:
                    best_bound = b
                    best_v = np.array([a0, a4, a8])
    
    return best_v, best_bound, A0, A2, AG


def construct():
    """
    Find coefficients (a0, a4, a8) that minimize the uncertainty bound.
    Returns np.array([a0, a4, a8]).
    """
    start_time = time.time()
    
    # Phase 1: Eigenvalue-based grid search to find initial good point
    try:
        best_v, best_bound, A0, A2, AG = _eigenvalue_approach()
    except Exception:
        best_v = np.array([-113.0/100, 1.0/25, 1.0/3240])
        best_bound = _compute_moments(best_v[0], best_v[1], best_v[2])
    
    if best_v is None:
        best_v = np.array([-113.0/100, 1.0/25, 1.0/3240])
        best_bound = _compute_moments(best_v[0], best_v[1], best_v[2])
    
    # Phase 2: Local refinement using Nelder-Mead from the best point
    if time.time() - start_time < _DEADLINE - 30:
        try:
            result = minimize(
                lambda x: _compute_moments(x[0], x[1], x[2]),
                best_v,
                method='Nelder-Mead',
                options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-18, 'adaptive': True}
            )
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_v = result.x.copy()
        except Exception:
            pass
    
    # Phase 3: Root-finding on stationary equations for higher precision
    if time.time() - start_time < _DEADLINE - 10:
        try:
            sol = root(
                _stationary_equations,
                best_v,
                method='hybr',
                options={'xtol': 1e-14, 'maxfev': 1000}
            )
            if sol.success or np.all(np.abs(sol.fun) < 1e-8):
                b = _compute_moments(sol.x[0], sol.x[1], sol.x[2])
                if np.isfinite(b) and b < best_bound:
                    best_bound = b
                    best_v = sol.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final Nelder-Mead polish
    if time.time() - start_time < _DEADLINE - 5:
        try:
            result = minimize(
                lambda x: _compute_moments(x[0], x[1], x[2]),
                best_v,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-16, 'fatol': 1e-20, 'adaptive': True}
            )
            if np.isfinite(result.fun) and result.fun < best_bound:
                best_bound = result.fun
                best_v = result.x.copy()
        except Exception:
            pass
    
    return best_v
# EVOLVE-BLOCK-END
