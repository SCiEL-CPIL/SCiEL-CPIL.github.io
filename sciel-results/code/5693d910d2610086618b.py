"""Search for coefficients minimizing r^2/(2*pi) where r is the largest
sign-changing real root of q = g/x^2, subject to g >= 0."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.special import hermite


def _hermite_coeffs(n):
    """Return coefficients of physicists' Hermite polynomial H_n(x) in ascending powers."""
    if n == 0:
        return np.array([1.0])
    if n == 1:
        return np.array([0.0, 2.0])
    
    h_prev = np.array([1.0])  # H_0
    h_curr = np.array([0.0, 2.0])  # H_1
    for k in range(2, n + 1):
        # H_k = 2x * H_{k-1} - 2(k-1) * H_{k-2}
        h_next = np.zeros(k + 1)
        h_next[1:] += 2.0 * h_curr
        h_next[:len(h_prev)] -= 2.0 * (k - 1) * h_prev
        h_prev, h_curr = h_curr, h_next
    return h_curr


# Precompute Hermite coefficients
_H0 = _hermite_coeffs(0)
_H4 = _hermite_coeffs(4)
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]
_H12_lead = _H12[-1]  # 2^12 = 4096


def _build_q(c0, c1, c2):
    """Return coefficients of q(x)=g(x)/x^2 (ascending powers) as a float array, and c3."""
    c0 = float(c0); c1 = float(c1); c2 = float(c2)
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    
    q = np.zeros(11)
    q[:] = g[2:]
    return q, c3


def _signchange_root(q):
    """Largest real root of q where q changes sign."""
    nz = np.nonzero(q)[0]
    if len(nz) == 0:
        return None
    q_trimmed = q[nz[0]:]
    roots = np.roots(q_trimmed)
    real = [z.real for z in roots if abs(z.imag) < 1e-9 * max(1.0, abs(z))]
    if not real:
        return None
    
    # Refine real roots using Newton's method to improve precision
    q_deriv = np.polyder(q_trimmed)
    refined_roots = []
    for r in real:
        # Check sign change with a small epsilon
        eps = 1e-8
        q_left = np.polyval(q, r - eps)
        q_right = np.polyval(q, r + eps)
        if q_left * q_right < 0:
            # Newton polish
            x = r
            for _ in range(5):
                fx = np.polyval(q_trimmed, x)
                dfx = np.polyval(q_deriv, x)
                if abs(dfx) < 1e-15:
                    break
                x_new = x - fx / dfx
                if abs(x_new - x) < 1e-12:
                    x = x_new
                    break
                x = x_new
            # Verify it's still a sign change
            eps_check = 1e-7
            ql = np.polyval(q, x - eps_check)
            qr = np.polyval(q, x + eps_check)
            if ql * qr < 0:
                refined_roots.append(x)
    
    if not refined_roots:
        return None
    return max(refined_roots)


def _eval_g(c0, c1, c2, xs, c3=None):
    """Evaluate g(x) at points xs."""
    if c3 is None:
        _, c3 = _build_q(c0, c1, c2)
    g = np.zeros(13)
    g[:len(_H0)] += c0 * _H0
    g[:len(_H4)] += c1 * _H4
    g[:len(_H8)] += c2 * _H8
    g[:len(_H12)] += c3 * _H12
    return np.polyval(g, xs)


def _objective(c):
    """Smooth objective with correct Hermite values."""
    q, c3 = _build_q(c[0], c[1], c[2])
    g_lead = c3 * _H12_lead

    r = _signchange_root(q)
    if r is None:
        r = 0.0
    obj = r * r

    # Check non-negativity on a grid
    xs = np.linspace(0.0, 20.0, 500)
    g = _eval_g(c[0], c[1], c[2], xs, c3)
    if g_lead < 0:
        g = -g
    mn = float(np.min(g))
    if mn < 0:
        # Strong penalty for constraint violation
        scale = max(1.0, abs(r), abs(mn))
        obj += 1000.0 * (mn / scale) ** 2
    return obj


def _value(c):
    """Exact value = r^2/(2*pi) for the sign-changing root; None if invalid."""
    q, c3 = _build_q(c[0], c[1], c[2])
    g_lead = c3 * _H12_lead
    r = _signchange_root(q)
    if r is None:
        return None
    
    # Check non-negativity: g(x) >= 0 for all x >= 0
    # Use a fine grid plus check at critical points (roots of g')
    xs = np.linspace(0.0, 30.0, 5000)
    g = _eval_g(c[0], c[1], c[2], xs, c3)
    if g_lead < 0:
        g = -g
    
    # Also check at roots of g' to catch local minima between grid points
    g_poly = np.zeros(13)
    g_poly[:len(_H0)] += c[0] * _H0
    g_poly[:len(_H4)] += c[1] * _H4
    g_poly[:len(_H8)] += c[2] * _H8
    g_poly[:len(_H12)] += c3 * _H12
    g_deriv = np.polyder(g_poly)
    if len(g_deriv) > 0:
        crit_roots = np.roots(g_deriv)
        real_crit = [z.real for z in crit_roots if abs(z.imag) < 1e-8 * max(1.0, abs(z)) and z.real >= 0]
        for cr in real_crit:
            g_cr = float(np.polyval(g_poly, cr))
            if g_lead < 0:
                g_cr = -g_cr
            if g_cr < -1e-8:
                return None
    
    min_g = float(np.min(g))
    if min_g < -1e-8:
        return None
        
    return r * r / (2.0 * np.pi)


def construct():
    t0 = time.time()
    deadline = t0 + 195.0

    seed = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    seed_val = _value(seed)
    if seed_val is None:
        seed_val = 1.0 / (2.0 * np.pi)
    best_c = seed.copy()
    best_val = seed_val

    def _try_update(c):
        nonlocal best_c, best_val
        v = _value(c)
        if v is not None and v < best_val:
            best_c = c.copy()
            best_val = v

    starts = [seed.copy()]
    rng = np.random.default_rng(12345)
    # More diverse starting points with wider spread
    for _ in range(40):
        starts.append(seed + rng.normal(0.0, 0.3, size=3))
    for s in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
        starts.append(np.array([-0.1, s, 0.02]))
        starts.append(np.array([0.1, -s, 0.02]))
        starts.append(np.array([-0.5, s * 0.5, 0.01]))
        starts.append(np.array([0.3, -s * 0.3, 0.015]))

    # Also try some structured perturbations of seed
    for delta in [0.01, 0.05, 0.1, 0.2, 0.5]:
        starts.append(seed + np.array([delta, 0, 0]))
        starts.append(seed + np.array([-delta, 0, 0]))
        starts.append(seed + np.array([0, delta, 0]))
        starts.append(seed + np.array([0, -delta, 0]))
        starts.append(seed + np.array([0, 0, delta]))
        starts.append(seed + np.array([0, 0, -delta]))

    for x0 in starts:
        if time.time() > deadline - 15.0:
            break
        try:
            # Coarse search with more iterations
            res = minimize(_objective, x0, method="Nelder-Mead",
                           options={"maxiter": 2000, "xatol": 1e-9, "fatol": 1e-11,
                                    "adaptive": True})
            _try_update(res.x)
            
            if time.time() > deadline - 15.0:
                break
            
            # Fine search (Polish)
            try:
                res2 = minimize(_objective, res.x, method="L-BFGS-B",
                                options={"maxiter": 2000, "gtol": 1e-13, "ftol": 1e-13})
                _try_update(res2.x)
            except Exception:
                pass
        except Exception:
            continue

    # Final high-precision polish on the best candidate found
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(_objective, best_c, method="L-BFGS-B",
                                 options={"maxiter": 5000, "gtol": 1e-14, "ftol": 1e-14})
            _try_update(res_final.x)
        except Exception:
            pass
        
        # One more Nelder-Mead polish from the best with tighter tolerance
        if time.time() < deadline - 3.0:
            try:
                res_nm = minimize(_objective, best_c, method="Nelder-Mead",
                                  options={"maxiter": 8000, "xatol": 1e-12, "fatol": 1e-14,
                                           "adaptive": True})
                _try_update(res_nm.x)
            except Exception:
                pass
        
        # Final verification and return
        final_val = _value(best_c)
        if final_val is None or final_val >= seed_val:
            return np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
        return np.array(best_c, dtype=float)


# EVOLVE-BLOCK-END
