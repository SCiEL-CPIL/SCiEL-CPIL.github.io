import numpy as np
import time
from scipy.optimize import minimize

def compute_value(coeffs):
    c0, c1, c2 = coeffs
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    A = -48.0 * c1 - 13440.0 * c2 - 11827200.0 * c3
    B = 16.0 * c1 + 11520.0 * c2 + 13977600.0 * c3
    C = -3072.0 * c2 - 6812160.0 * c3
    D = 256.0 * c2 + 1397760.0 * c3
    E = -122880.0 * c3
    F = 4096.0 * c3
    
    coeffs_q = [F, E, D, C, B, A]
    
    roots = np.roots(coeffs_q)
    
    valid_roots = []
    for root in roots:
        if abs(root.imag) < 1e-10:
            r = root.real
            if r > 1e-12:
                eps = max(1e-8, abs(r) * 1e-8)
                q_minus = np.polyval(coeffs_q, r - eps)
                q_plus = np.polyval(coeffs_q, r + eps)
                if q_minus * q_plus < 0:
                    valid_roots.append(r)
            
    if not valid_roots:
        return 1e9
        
    r_max = max(valid_roots)
    return r_max ** 2 / (2.0 * np.pi)

def _g_values_fast(c0, c1, c2, x_grid):
    c3 = -(c0 * 1.0 + c1 * 12.0 + c2 * 1680.0) / 4084080.0
    
    x2 = x_grid ** 2
    x4 = x2 ** 2
    x6 = x4 * x2
    x8 = x4 ** 2
    x10 = x8 * x2
    x12 = x8 * x4
    
    g = (c0 * 1.0 +
         c1 * (16.0 * x4 - 48.0 * x2 + 12.0) +
         c2 * (256.0 * x8 - 3072.0 * x6 + 11520.0 * x4 - 13440.0 * x2 + 1680.0) +
         c3 * (4096.0 * x12 - 122880.0 * x10 + 1397760.0 * x8 - 6812160.0 * x6 + 13977600.0 * x4 - 11827200.0 * x2 + 4084080.0))
    return g

def _soft_min_penalty(g_vals, scale=50.0):
    neg_g = -g_vals
    neg_g_clipped = np.clip(neg_g, -50.0, 50.0)
    max_neg = np.max(neg_g_clipped)
    if max_neg < -50.0:
        return 0.0
    sum_exp = np.sum(np.exp(neg_g_clipped - max_neg))
    penalty = (1.0 / scale) * (max_neg + np.log(sum_exp))
    return max(0.0, penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 175.0
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    x_grid = np.linspace(0.001, 10.0, 1000)
    
    def objective(params):
        c0, c1, c2 = params
        val = compute_value([c0, c1, c2])
        if val >= 1e8:
            return 1e8
        g_vals = _g_values_fast(c0, c1, c2, x_grid)
        penalty = _soft_min_penalty(g_vals, scale=50.0)
        return val + 10.0 * penalty
    
    rng = np.random.default_rng(42)
    
    start_points = [seed.copy()]
    for _ in range(15):
        start_points.append(seed + rng.normal(0, 0.3, size=3))
    for _ in range(25):
        start_points.append(rng.uniform(-2.0, 2.0, size=3))
    for _ in range(10):
        start_points.append(seed + rng.normal(0, 1.0, size=3))
    
    for i, start in enumerate(start_points):
        if time.time() > deadline - 25.0:
            break
        
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 100, 'xatol': 1e-9, 'fatol': 1e-9,
                                    'adaptive': True})
            
            if res.fun < best_val:
                x_fine = np.linspace(0.001, 10.0, 3000)
                g_check = _g_values_fast(res.x[0], res.x[1], res.x[2], x_fine)
                if np.min(g_check) >= -1e-6:
                    actual_val = compute_value(res.x)
                    if actual_val < best_val:
                        best_val = actual_val
                        best_coeffs = res.x.copy()
        except Exception:
            pass
    
    if best_val < seed_val and time.time() < deadline - 15.0:
        try:
            x_fine = np.linspace(0.001, 10.0, 5000)
            
            def objective_precise(params):
                c0, c1, c2 = params
                val = compute_value([c0, c1, c2])
                if val >= 1e8:
                    return 1e8
                g_vals = _g_values_fast(c0, c1, c2, x_fine)
                penalty = _soft_min_penalty(g_vals, scale=100.0)
                return val + 5.0 * penalty
            
            res_final = minimize(objective_precise, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 300, 'xatol': 1e-12, 'fatol': 1e-12,
                                          'adaptive': True})
            g_final = _g_values_fast(res_final.x[0], res_final.x[1], res_final.x[2], x_fine)
            if np.min(g_final) >= -1e-7:
                actual_final = compute_value(res_final.x)
                if actual_final < best_val:
                    best_val = actual_final
                    best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    if best_val >= seed_val:
        return seed
    
    return best_coeffs
