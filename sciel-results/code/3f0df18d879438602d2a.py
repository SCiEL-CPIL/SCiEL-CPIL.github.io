"""Optimized program: differential evolution + Nelder-Mead polish for minimal sign-changing root."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline guard
_START_TIME = time.time()
TIME_LIMIT = 200.0  # safety margin before 210s kill

def _time_left():
    return TIME_LIMIT - (time.time() - _START_TIME)

# Precompute Hermite polynomial coefficients (physicists' convention)
# H_n(x) = n! * sum_{k=0}^{floor(n/2)} (-1)^k * (2x)^(n-2k) / (k! * (n-2k)!)
# We need coefficients of x^m for m = 0, 2, 4, 6, 8, 10, 12 (even powers only)

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn]."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]  # H_{k-1}
        prev2 = H[k - 2]  # H_{k-2}
        curr = [0.0] * (k + 1)
        # 2x * H_{k-1}: shift prev1 by 1 and multiply by 2
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        # -2(k-1) * H_{k-2}
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

# Precompute for H0, H4, H8, H12
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Values at x=0
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def compute_value(c0, c1, c2):
    """Compute value = r^2 / (2*pi) for given coefficients."""
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g(x) coefficients (ascending power order)
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for coeff, H in [(c0, _H0), (c1, _H4), (c2, _H8)]:
        for i in range(len(H)):
            g_coeffs[i] += coeff * H[i]
    
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    # Check if g tends to negative at infinity
    lead_coeff = g_coeffs[12]
    if lead_coeff < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    # q = g / x^2
    if abs(g_coeffs[0]) > 1e-15 or abs(g_coeffs[1]) > 1e-15:
        return float('inf')
    
    q_coeffs = g_coeffs[2:]  # Divide by x^2, now degree 10
    
    # q(x) = q0 + q2*x^2 + q4*x^4 + q6*x^6 + q8*x^8 + q10*x^10
    # Let z = x^2: Q(z) = q10*z^5 + q8*z^4 + q6*z^3 + q4*z^2 + q2*z + q0
    q0 = q_coeffs[0]
    q2 = q_coeffs[2]
    q4 = q_coeffs[4]
    q6 = q_coeffs[6]
    q8 = q_coeffs[8]
    q10 = q_coeffs[10]
    
    # Q(z) coefficients in descending order for np.roots
    z_coeffs = np.array([q10, q8, q6, q4, q2, q0])
    
    # Check if polynomial is essentially zero
    if np.allclose(z_coeffs, 0, atol=1e-30):
        return np.inf
    
    roots = np.roots(z_coeffs)
    
    # Filter: real, positive
    valid_roots = []
    for rt in roots:
        if abs(rt.imag) < 1e-10 and rt.real > 1e-15:
            z_val = rt.real
            # Check sign change: Q(z) changes sign at z_val
            eps = 1e-8 * max(1.0, abs(z_val))
            Q_minus = q10*(z_val-eps)**5 + q8*(z_val-eps)**4 + q6*(z_val-eps)**3 + q4*(z_val-eps)**2 + q2*(z_val-eps) + q0
            Q_plus = q10*(z_val+eps)**5 + q8*(z_val+eps)**4 + q6*(z_val+eps)**3 + q4*(z_val+eps)**2 + q2*(z_val+eps) + q0
            if Q_minus * Q_plus < 0:
                valid_roots.append(z_val)
    
    if not valid_roots:
        return np.inf
    
    # r = largest real root where q changes sign
    # value = r^2 / (2*pi) = z_val / (2*pi)
    r_max = np.sqrt(max(valid_roots))
    return r_max**2 / (2 * np.pi)

def objective(params):
    """Objective function for optimization."""
    c0, c1, c2 = params
    val = compute_value(c0, c1, c2)
    if np.isinf(val) or np.isnan(val):
        return 1e10
    return val

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    seed = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = compute_value(seed[0], seed[1], seed[2])
    best_params = seed.copy()
    
    # Phase 1: Differential Evolution for global exploration
    if _time_left() > 120:
        try:
            bounds = [(-5, 5), (-5, 5), (-2, 2)]
            de_result = differential_evolution(
                objective, bounds,
                seed=42,
                maxiter=100,
                popsize=20,
                tol=1e-8,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if np.isfinite(de_result.fun) and de_result.fun < best_val:
                best_val = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase 2: Multi-start Nelder-Mead for local refinement
    rng = np.random.RandomState(42)
    
    starts = [best_params.copy()]
    
    # Add diverse starting points
    for i in range(15):
        scale = 0.3 * (i + 1) / 15
        perturbation = rng.randn(3) * scale
        starts.append(best_params + perturbation)
    
    # Add wider exploration
    for i in range(10):
        scale = 1.0 + 1.5 * i / 10
        perturbation = rng.randn(3) * scale
        starts.append(best_params + perturbation)
    
    nm_options = {
        'xatol': 1e-10,
        'fatol': 1e-12,
        'maxiter': 500,
        'maxfev': 2000,
        'adaptive': True,
    }
    
    for start in starts:
        if _time_left() < 30:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead', options=nm_options)
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 3: Final polish with tighter tolerance
    if _time_left() > 30:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 1000, 'maxfev': 5000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 4: Ultra-fine polish
    if _time_left() > 20:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                          options={'xatol': 1e-13, 'fatol': 1e-15, 'maxiter': 2000, 'maxfev': 10000, 'adaptive': True})
            if np.isfinite(res.fun) and res.fun < best_val:
                best_val = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Safety: if time is running out, return best found
    if _time_left() < 5:
        return best_params
    
    # Verify result
    final_val = compute_value(best_params[0], best_params[1], best_params[2])
    if np.isinf(final_val) or np.isnan(final_val) or final_val > best_val * 1.01:
        # Fallback to seed
        return seed
    
    return best_params
# EVOLVE-BLOCK-END
