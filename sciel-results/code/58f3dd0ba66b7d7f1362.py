"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_hermite  # not used directly, but available
import sympy as sp

# Precompute Hermite polynomial coefficients as numpy arrays for speed
# H_n(x) from sympy.polys.orthopolys.hermite_poly
# We need H0, H4, H8, H12

def _hermite_coeffs(n):
    """Return coefficients of physicist's Hermite polynomial H_n as numpy array [a0, a1, ..., an]
    where H_n(x) = a0 + a1*x + ... + an*x^n."""
    H = sp.hermite_poly(n)
    coeffs = sp.Poly(H, sp.Symbol('x')).all_coeffs()  # highest degree first
    # Convert to ascending order
    coeffs = list(reversed(coeffs))
    return np.array([float(c) for c in coeffs])

# Precompute
_H0 = _hermite_coeffs(0)   # [1]
_H4 = _hermite_coeffs(4)   # [24, 0, -12, 0, 1]
_H8 = _hermite_coeffs(8)
_H12 = _hermite_coeffs(12)

# Values at x=0
_H0_0 = float(sp.hermite_poly(0).subs(sp.Symbol('x'), 0))  # 1
_H4_0 = float(sp.hermite_poly(4).subs(sp.Symbol('x'), 0))  # 24
_H8_0 = float(sp.hermite_poly(8).subs(sp.Symbol('x'), 0))
_H12_0 = float(sp.hermite_poly(12).subs(sp.Symbol('x'), 0))

# Build g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
# g(0) = 0 => c3 = -(c0*H0(0) + c1*H4(0) + c2*H8(0)) / H12(0)
# q(x) = g(x) / x^2, which must divide exactly (g has x^2 factor since g(0)=0 and we need divisibility)
# Actually g(0)=0 ensures at least one factor of x. We need x^2 divides g.
# Let's verify: H0(0)=1, H4(0)=24, H8(0)=1344, H12(0)=1032192
# g(0) = c0*1 + c1*24 + c2*1344 + c3*1032192 = 0
# g'(0): H0'(0)=0, H4'(0)=0, H8'(0)=0, H12'(0)=0 (all even polynomials, odd derivatives at 0 are 0)
# So g'(0) = 0 automatically! Thus x^2 always divides g. Good.

def _build_q_coeffs(c0, c1, c2):
    """Build coefficients of q(x) = g(x)/x^2 as numpy array in ascending order."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # g(x) = c0*H0 + c1*H4 + c2*H8 + c3*H12
    # Build g coefficients (ascending order)
    max_deg = 12
    g_coeffs = np.zeros(max_deg + 1)
    
    for i in range(len(_H0)):
        g_coeffs[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g_coeffs[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g_coeffs[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    # q(x) = g(x)/x^2, so remove first two coefficients
    # g(0) should be 0, g'(0) should be 0
    # q_coeffs = g_coeffs[2:]
    q_coeffs = g_coeffs[2:]
    return q_coeffs

def _eval_poly(coeffs, x):
    """Evaluate polynomial with ascending coefficients at x."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result

def _find_sign_changing_roots(q_coeffs, x_min=-20.0, x_max=20.0, n_samples=2000):
    """Find all sign-changing roots of q in [x_min, x_max]."""
    # Sample q on a grid
    xs = np.linspace(x_min, x_max, n_samples)
    ys = np.array([_eval_poly(q_coeffs, x) for x in xs])
    
    roots = []
    for i in range(len(xs) - 1):
        if ys[i] * ys[i+1] < 0:
            # Sign change, bisect
            a, b = xs[i], xs[i+1]
            fa, fb = ys[i], ys[i+1]
            for _ in range(60):
                m = 0.5 * (a + b)
                fm = _eval_poly(q_coeffs, m)
                if fa * fm <= 0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
            root = 0.5 * (a + b)
            if root > 1e-10:  # Ignore root at 0 (we divided by x^2, but check)
                roots.append(root)
    
    return roots

def _objective(c0, c1, c2, beta=100.0):
    """Smoothed objective: soft-max over sign-changing roots of r^2/(2*pi).
    
    Uses log-sum-exp to smooth the 'largest root' selection.
    """
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    roots = _find_sign_changing_roots(q_coeffs)
    
    if not roots:
        return 1e10  # Large penalty if no sign-changing roots
    
    # Filter positive roots (r should be positive for r^2)
    positive_roots = [r for r in roots if r > 1e-8]
    
    if not positive_roots:
        return 1e10
    
    # Soft-max: log-sum-exp over log(r^2) = 2*log(r)
    # value = r^2 / (2*pi)
    # We want to minimize the max, so use log-sum-exp as smooth upper bound
    log_r2 = [2.0 * np.log(r) for r in positive_roots]
    max_log_r2 = max(log_r2)
    
    # log-sum-exp
    if len(log_r2) == 1:
        log_sum_exp = log_r2[0]
    else:
        # log(sum(exp(log_r2))) = max + log(sum(exp(log_r2 - max)))
        shifted = [lr - max_log_r2 for lr in log_r2]
        log_sum_exp = max_log_r2 + np.log(sum(np.exp(s) for s in shifted))
    
    # The smooth value is exp(log_sum_exp) / (2*pi)
    # But this is an upper bound on max(r^2)/(2*pi)
    # As beta -> inf, it approaches the true max
    # For finite beta, it's a smooth surrogate
    value = np.exp(log_sum_exp) / (2.0 * np.pi)
    
    return value

def _objective_exact(c0, c1, c2):
    """Exact objective (nonsmooth): largest sign-changing root squared / (2*pi)."""
    q_coeffs = _build_q_coeffs(c0, c1, c2)
    roots = _find_sign_changing_roots(q_coeffs)
    
    if not roots:
        return 1e10
    
    positive_roots = [r for r in roots if r > 1e-8]
    
    if not positive_roots:
        return 1e10
    
    r_max = max(positive_roots)
    return r_max ** 2 / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 180.0  # Safety margin before 210s kill
    
    best_value = 1e10
    best_coeffs = np.array([-113.0 / 100.0, 1.0 / 25.0, 1.0 / 3240.0])
    
    # Get seed value
    seed_value = _objective_exact(best_coeffs[0], best_coeffs[1], best_coeffs[2])
    best_value = min(best_value, seed_value)
    
    # Multi-start local optimization with smoothed objective
    # Generate diverse starting points
    starts = []
    
    # Seed
    starts.append(best_coeffs.copy())
    
    # Grid of perturbations
    for dc0 in [-0.1, -0.05, 0.0, 0.05, 0.1]:
        for dc1 in [-0.02, -0.01, 0.0, 0.01, 0.02]:
            for dc2 in [-0.001, -0.0005, 0.0, 0.0005, 0.001]:
                if dc0 == 0 and dc1 == 0 and dc2 == 0:
                    continue
                start = best_coeffs + np.array([dc0, dc1, dc2])
                starts.append(start)
    
    # Random starts
    rng = np.random.default_rng(42)
    for _ in range(30):
        rand_start = best_coeffs + rng.normal(0, 0.05, size=3)
        starts.append(rand_start)
    
    # Scale up/down
    for scale in [0.5, 0.8, 1.2, 1.5, 2.0]:
        starts.append(best_coeffs * scale)
    
    # Local optimization from each start
    for i, start in enumerate(starts):
        if time.time() > deadline - 30.0:  # Leave 30s for final polish
            break
        
        try:
            result = minimize(
                lambda x: _objective(x[0], x[1], x[2], beta=100.0),
                start,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.success or result.fun < best_value:
                val = _objective_exact(result.x[0], result.x[1], result.x[2])
                if val < best_value:
                    best_value = val
                    best_coeffs = result.x.copy()
        except Exception:
            continue
    
    # Final high-precision polish with COBYLA on exact objective
    if time.time() < deadline - 10.0:
        try:
            result = minimize(
                lambda x: _objective_exact(x[0], x[1], x[2]),
                best_coeffs,
                method='COBYLA',
                options={'maxiter': 200, 'rhobeg': 1e-6, 'tol': 1e-12}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    # One more polish if time permits
    if time.time() < deadline - 5.0:
        try:
            result = minimize(
                lambda x: _objective_exact(x[0], x[1], x[2]),
                best_coeffs,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True}
            )
            
            if result.fun < best_value:
                best_value = result.fun
                best_coeffs = result.x.copy()
        except Exception:
            pass
    
    return best_coeffs
# EVOLVE-BLOCK-END
