import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n."""
    # H0 = 1, H1 = 2x, H_{n+1} = 2x*Hn - 2n*H_{n-1}
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

# Precompute Hermite coefficients
_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

# Values at x=0
_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _compute_value(c0, c1, c2):
    """Compute the value for given coefficients."""
    # Build g = c0*H0 + c1*H4 + c2*H8 + c3*H12
    max_deg = 12
    g_coeffs = [0.0] * (max_deg + 1)
    
    for coeff, H in [(c0, _H0), (c1, _H4), (c2, _H8)]:
        for i in range(len(H)):
            g_coeffs[i] += coeff * H[i]
    
    # Compute c3 so that g(0) = 0
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    for i in range(len(_H12)):
        g_coeffs[i] += c3 * _H12[i]
    
    # Check if g tends to negative at infinity
    lead_coeff = g_coeffs[12]
    if lead_coeff < 0:
        g_coeffs = [-c for c in g_coeffs]
    
    # q = g / x^2
    if abs(g_coeffs[0]) > 1e-15 or abs(g_coeffs[1]) > 1e-15:
        return float('inf')
    
    q_coeffs = g_coeffs[2:]  # Divide by x^2
    
    # Find largest sign-changing root
    if len(q_coeffs) == 0:
        return float('inf')
    
    coeffs_desc = list(reversed(q_coeffs))
    roots = np.roots(coeffs_desc)
    
    # Filter real roots
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            real_roots.append(r.real)
    
    # For each positive real root, check if sign changes
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in real_roots:
        if r <= 0:
            continue
        # Check sign change: evaluate at r - eps and r + eps
        eps = 1e-8 * max(1.0, abs(r))
        val_left = poly(r - eps)
        val_right = poly(r + eps)
        if val_left * val_right < 0:
            # Sign change at r
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _compute_value_fast(c0, c1, c2):
    """Faster version using precomputed values where possible."""
    # g(0) = 0 constraint gives c3
    c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
    
    # Build g coefficients
    g = [0.0] * 13
    for i in range(len(_H0)):
        g[i] += c0 * _H0[i]
    for i in range(len(_H4)):
        g[i] += c1 * _H4[i]
    for i in range(len(_H8)):
        g[i] += c2 * _H8[i]
    for i in range(len(_H12)):
        g[i] += c3 * _H12[i]
    
    # Check leading coefficient
    if g[12] < 0:
        g = [-c for c in g]
    
    # q = g / x^2
    if abs(g[0]) > 1e-15 or abs(g[1]) > 1e-15:
        return float('inf')
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10 and r.real > 0]
    if not real_roots:
        return float('inf')
    
    poly = np.poly1d(coeffs_desc)
    largest = None
    for r in real_roots:
        eps = 1e-8 * max(1.0, abs(r))
        if poly(r - eps) * poly(r + eps) < 0:
            if largest is None or r > largest:
                largest = r
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _smooth_objective(x, beta=20.0):
    """Smooth surrogate using log-sum-exp over sign-changing roots.
    
    For each sign-changing root r_i, we have a 'cost' of r_i^2 / (2*pi).
    The true objective is max_i cost_i. We approximate with:
        (1/beta) * log(sum_i exp(beta * cost_i))
    which converges to max as beta -> inf but is differentiable.
    """
    c0, c1, c2 = x
    try:
        c3 = -(c0 * _H0_0 + c1 * _H4_0 + c2 * _H8_0) / _H12_0
        
        g = [0.0] * 13
        for i in range(len(_H0)):
            g[i] += c0 * _H0[i]
        for i in range(len(_H4)):
            g[i] += c1 * _H4[i]
        for i in range(len(_H8)):
            g[i] += c2 * _H8[i]
        for i in range(len(_H12)):
            g[i] += c3 * _H12[i]
        
        if g[12] < 0:
            g = [-c for c in g]
        
        if abs(g[0]) > 1e-15 or abs(g[1]) > 1e-15:
            return 1e10
        
        q = g[2:]
        coeffs_desc = list(reversed(q))
        roots = np.roots(coeffs_desc)
        
        sign_changing_costs = []
        poly = np.poly1d(coeffs_desc)
        
        for r in roots:
            if abs(r.imag) < 1e-10 and r.real > 0:
                rr = r.real
                eps = 1e-8 * max(1.0, abs(rr))
                if poly(rr - eps) * poly(rr + eps) < 0:
                    cost = rr ** 2 / (2.0 * np.pi)
                    sign_changing_costs.append(cost)
        
        if not sign_changing_costs:
            return 1e10
        
        costs = np.array(sign_changing_costs)
        # Log-sum-exp for numerical stability
        max_cost = np.max(costs)
        log_sum_exp = max_cost + np.log(np.sum(np.exp(beta * (costs - max_cost))))
        return log_sum_exp / beta
    except:
        return 1e10

def construct():
    start_time = time.time()
    deadline = 195.0  # Safety margin before 210s
    
    # Seed
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(x):
        try:
            return _compute_value_fast(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Phase 1: Differential evolution for global search
    if time.time() < deadline - 60:
        try:
            bounds = [(-5, 5), (-5, 5), (-5, 5)]
            de_result = differential_evolution(
                objective, bounds, maxiter=100, popsize=15,
                tol=1e-8, seed=42, polish=False
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Multi-start Nelder-Mead for local refinement
    np.random.seed(42)
    num_starts = 30
    for i in range(num_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x0 = best_c.copy()
        else:
            # Sample around best and random
            if i % 3 == 0:
                x0 = best_c + np.random.normal(0, 0.5, size=3)
            elif i % 3 == 1:
                x0 = np.random.uniform(-3, 3, size=3)
            else:
                x0 = best_c + np.random.normal(0, 0.1, size=3)
        
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Phase 3: Smooth surrogate refinement with log-sum-exp
    if time.time() < deadline - 20:
        for beta in [10.0, 20.0, 50.0, 100.0]:
            if time.time() > deadline - 15:
                break
            try:
                result = minimize(lambda x: _smooth_objective(x, beta), best_c,
                                 method='Nelder-Mead',
                                 options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
                # Verify with exact objective
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
            except:
                pass
    
    # Phase 4: High-precision polish with tighter tolerance
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-15, 'fatol': 1e-15, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
        except:
            pass
    
    # Final verification
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val
    
    return best_c
