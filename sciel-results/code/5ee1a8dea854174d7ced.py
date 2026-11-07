"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _compute_value(c0, c1, c2):
    """Compute the value r^2/(2*pi) for given coefficients."""
    # Hermite polynomials (physicists' convention)
    # H0(x) = 1
    # H4(x) = 16x^4 - 48x^2 + 12
    # H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1344
    # H12(x) = 4096x^12 - 122880x^10 + 1290240x^8 - 5806080x^6 + 9953280x^4 - 5505024x^2 + 1032192
    
    # Values at x=0
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1344.0
    H12_0 = 1032192.0
    
    # c3 is determined by g(0) = 0
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # q(x) = g(x)/x^2. Since g(0)=0 and g is even, g(x) = x^2 * q(x) where q is even.
    # Let t = x^2, then q(x) = Q(t) where Q is a polynomial in t.
    
    # Coefficients of Q(t) = a5*t^5 + a4*t^4 + a3*t^3 + a2*t^2 + a1*t + a0
    a0 = -48*c1 - 13440*c2 - 5505024*c3
    a1 = 16*c1 + 11520*c2 + 9953280*c3
    a2 = -3072*c2 - 5806080*c3
    a3 = 256*c2 + 1290240*c3
    a4 = -122880*c3
    a5 = 4096*c3
    
    # We need the largest real root r of q(x) = Q(x^2) where q changes sign.
    # Since q(x) = Q(x^2), roots of q correspond to roots of Q at t = r^2 > 0.
    # q changes sign at x=r iff Q changes sign at t=r^2.
    
    # Find all real roots of Q(t) for t > 0
    coeffs_Q = [a5, a4, a3, a2, a1, a0]
    
    # Use numpy to find roots
    try:
        roots_t = np.roots(coeffs_Q)
    except:
        return float('inf')
    
    # Filter for real positive roots
    real_roots = []
    for rt in roots_t:
        if abs(rt.imag) < 1e-10:
            tr = rt.real
            if tr > 1e-12:  # positive
                real_roots.append(tr)
    
    if not real_roots:
        return float('inf')
    
    # For each positive real root t_i, check if Q changes sign there
    best_r2 = float('inf')
    for tr in real_roots:
        # Check sign change: evaluate Q at tr - eps and tr + eps
        eps = 1e-8 * max(1.0, abs(tr))
        Q_minus = np.polyval(coeffs_Q, tr - eps)
        Q_plus = np.polyval(coeffs_Q, tr + eps)
        if Q_minus * Q_plus < 0:
            # Sign change, this is a valid root
            r2 = tr
            if r2 < best_r2:
                best_r2 = r2
    
    if best_r2 == float('inf'):
        return float('inf')
    
    return best_r2 / (2.0 * np.pi)


def _seed_value():
    """Compute the value for the seed coefficients."""
    c0, c1, c2 = -113/100, 1/25, 1/3240
    return _compute_value(c0, c1, c2)


def construct():
    # Prior state of the art (Goncalves et al. 2017) as listed in the notebook.
    seed = np.array([-113 / 100, 1 / 25, 1 / 3240])
    
    start_time = time.time()
    deadline = start_time + 195  # safety margin before 210s limit
    
    best_coeffs = seed.copy()
    best_val = _compute_value(seed[0], seed[1], seed[2])
    
    # Local optimization using Nelder-Mead or similar
    def objective(x):
        return _compute_value(x[0], x[1], x[2])
    
    # Try multiple optimization strategies
    for method in ['Nelder-Mead', 'Powell', 'COBYLA']:
        if time.time() > deadline:
            break
        try:
            result = minimize(
                objective,
                best_coeffs,
                method=method,
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    # Refine with more iterations if time permits
    if time.time() < deadline - 20:
        try:
            result = minimize(
                objective,
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
            )
            if result.fun < best_val:
                best_val = result.fun
                best_coeffs = result.x
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
