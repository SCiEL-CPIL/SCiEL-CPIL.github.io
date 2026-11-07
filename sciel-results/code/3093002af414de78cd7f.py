import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _get_hermite_coeffs(n):
    """Get coefficients of physicists' Hermite polynomial Hn(x) as list [c0, c1, ..., cn] for c0 + c1*x + ... + cn*x^n."""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    
    H = [[1.0], [0.0, 2.0]]
    for k in range(2, n + 1):
        prev1 = H[k - 1]
        prev2 = H[k - 2]
        curr = [0.0] * (k + 1)
        for i in range(k):
            curr[i + 1] += 2.0 * prev1[i]
        for i in range(k - 1):
            curr[i] += -2.0 * (k - 1) * prev2[i]
        H.append(curr)
    return H[n]

_H0 = _get_hermite_coeffs(0)
_H4 = _get_hermite_coeffs(4)
_H8 = _get_hermite_coeffs(8)
_H12 = _get_hermite_coeffs(12)

_H0_0 = _H0[0]
_H4_0 = _H4[0]
_H8_0 = _H8[0]
_H12_0 = _H12[0]

def _compute_value(c0, c1, c2):
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
        return float('inf')
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            rr = r.real
            eps = 1e-8 * max(1.0, abs(rr))
            if poly(rr - eps) * poly(rr + eps) < 0:
                if largest is None or rr > largest:
                    largest = rr
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _compute_value_fast(c0, c1, c2):
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
        return float('inf')
    
    q = g[2:]
    coeffs_desc = list(reversed(q))
    roots = np.roots(coeffs_desc)
    poly = np.poly1d(coeffs_desc)
    
    largest = None
    for r in roots:
        if abs(r.imag) < 1e-10 and r.real > 0:
            rr = r.real
            eps = 1e-8 * max(1.0, abs(rr))
            if poly(rr - eps) * poly(rr + eps) < 0:
                if largest is None or rr > largest:
                    largest = rr
    
    if largest is None:
        return float('inf')
    
    return largest ** 2 / (2 * np.pi)

def _smooth_objective(x, beta=50.0):
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
        poly = np.poly1d(coeffs_desc)
        
        sign_changing_costs = []
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
        max_cost = np.max(costs)
        log_sum_exp = max_cost + np.log(np.sum(np.exp(beta * (costs - max_cost))))
        return log_sum_exp / beta
    except:
        return 1e10

def _coordinate_descent(x, objective, max_iter=5000, tol=1e-14):
    """Coordinate descent with adaptive step size."""
    x = x.copy()
    best_f = objective(x)
    step = 0.1
    
    for it in range(max_iter):
        improved = False
        for i in range(3):
            for direction in [-1, 1]:
                x_try = x.copy()
                x_try[i] += direction * step
                f_try = objective(x_try)
                if f_try < best_f:
                    x = x_try
                    best_f = f_try
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.5
            if step < tol:
                break
        else:
            step = min(step * 1.2, 0.5)
    
    return x, best_f

def construct():
    start_time = time.time()
    deadline = 195.0
    
    best_c = np.array([-113.0 / 100, 1.0 / 25, 1.0 / 3240])
    best_val = _compute_value(best_c[0], best_c[1], best_c[2])
    
    def objective(x):
        try:
            return _compute_value_fast(x[0], x[1], x[2])
        except:
            return 1e10
    
    # Phase 1: Focused differential evolution around seed
    if time.time() < deadline - 80:
        try:
            bounds = [(-3, 3), (-3, 3), (-3, 3)]
            de_result = differential_evolution(
                objective, bounds, maxiter=150, popsize=20,
                tol=1e-10, seed=42, polish=False,
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_c = de_result.x.copy()
        except:
            pass
    
    # Phase 2: Multi-start with coordinate descent polish
    np.random.seed(42)
    num_starts = 40
    for i in range(num_starts):
        if time.time() > deadline - 40:
            break
        
        if i == 0:
            x0 = best_c.copy()
        elif i < 10:
            x0 = best_c + np.random.normal(0, 0.3, size=3)
        elif i < 20:
            x0 = best_c + np.random.normal(0, 0.1, size=3)
        elif i < 30:
            x0 = np.random.uniform(-2, 2, size=3)
        else:
            x0 = best_c + np.random.normal(0, 0.05, size=3)
        
        try:
            # Nelder-Mead first
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-13, 'fatol': 1e-13, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
            
            # Coordinate descent polish
            cd_x, cd_f = _coordinate_descent(best_c, objective, max_iter=3000)
            if cd_f < best_val:
                best_val = cd_f
                best_c = cd_x.copy()
        except:
            pass
    
    # Phase 3: Smooth surrogate with multiple betas
    if time.time() < deadline - 25:
        for beta in [20.0, 50.0, 100.0, 200.0]:
            if time.time() > deadline - 20:
                break
            try:
                result = minimize(lambda x: _smooth_objective(x, beta), best_c,
                                 method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14, 'adaptive': True})
                val = objective(result.x)
                if val < best_val:
                    best_val = val
                    best_c = result.x.copy()
                
                # Polish with coordinate descent
                cd_x, cd_f = _coordinate_descent(best_c, objective, max_iter=5000)
                if cd_f < best_val:
                    best_val = cd_f
                    best_c = cd_x.copy()
            except:
                pass
    
    # Phase 4: Ultra-precise final polish
    if time.time() < deadline - 5:
        try:
            result = minimize(objective, best_c, method='Nelder-Mead',
                             options={'maxiter': 20000, 'xatol': 1e-16, 'fatol': 1e-16, 'adaptive': True})
            if result.fun < best_val:
                best_val = result.fun
                best_c = result.x.copy()
            
            cd_x, cd_f = _coordinate_descent(best_c, objective, max_iter=10000, tol=1e-15)
            if cd_f < best_val:
                best_val = cd_f
                best_c = cd_x.copy()
        except:
            pass
    
    final_val = _compute_value(best_c[0], best_c[1], best_c[2])
    if final_val < best_val:
        best_val = final_val
    
    return best_c
