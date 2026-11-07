import numpy as np
import time
from scipy.optimize import minimize

# Precompute Hermite polynomial coefficients for H0, H4, H8, H12
# H0(x) = 1
# H4(x) = 16x^4 - 48x^2 + 12
# H8(x) = 256x^8 - 3072x^6 + 11520x^4 - 13440x^2 + 1680
# H12(x) = 4096x^12 - 122880x^10 + 1397760x^8 - 6812160x^6 + 13977600x^4 - 11827200x^2 + 4084080

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
    
    # Find roots of q(y)
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
    """Evaluate g(x) at points in x_grid. Returns array."""
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
    """Smooth approximation of max(0, -min(g_vals)) using log-sum-exp."""
    # We want a smooth penalty for min(g_vals) < 0
    # Use: penalty = (1/scale) * log(1 + sum(exp(-scale * g_vals)))
    # This is ~0 when all g_vals >> 0, and ~ -min(g_vals) when min is very negative
    neg_g = -g_vals
    # Clip to avoid overflow
    neg_g_clipped = np.clip(neg_g, -50.0, 50.0)
    max_neg = np.max(neg_g_clipped)
    if max_neg < -50.0:
        return 0.0
    sum_exp = np.sum(np.exp(neg_g_clipped - max_neg))
    penalty = (1.0 / scale) * (max_neg + np.log(sum_exp))
    return max(0.0, penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Grid for positivity check - use a reasonable range
    # The relevant x range is roughly [0, sqrt(r_max)] where r_max is the largest root
    # For the seed, r_max is around 1-2, so x_max ~ 1.5
    # Use a grid that covers [0, 5] with good resolution
    x_max_grid = 5.0
    n_grid = 500
    x_grid = np.linspace(0.001, x_max_grid, n_grid)  # skip x=0
    
    def objective(params):
        """Smooth penalty-based objective."""
        c0, c1, c2 = params
        
        # Compute the actual value
        val = compute_value([c0, c1, c2])
        if val >= 1e8:
            return 1e8
        
        # Positivity penalty using soft-min
        g_vals = _g_values_fast(c0, c1, c2, x_grid)
        penalty = _soft_min_penalty(g_vals, scale=50.0)
        
        return val + 10.0 * penalty
    
    # Multi-start optimization with Nelder-Mead and SLSQP
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    start_points = [seed.copy()]
    
    # Perturbed seed
    for _ in range(10):
        start_points.append(seed + rng.normal(0, 0.5, size=3))
    
    # Random points in a reasonable range
    for _ in range(20):
        start_points.append(rng.uniform(-3.0, 3.0, size=3))
    
    # Points near the boundary of feasibility
    for _ in range(10):
        start_points.append(seed + rng.normal(0, 1.5, size=3))
    
    n_evals = 0
    for i, start in enumerate(start_points):
        if time.time() > deadline - 20.0:
            break
        
        # Use Nelder-Mead for robustness
        try:
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 150, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            n_evals += 1
            
            if res.fun < best_val:
                # Verify with a finer grid
                x_fine = np.linspace(0.001, 5.0, 2000)
                g_check = _g_values_fast(res.x[0], res.x[1], res.x[2], x_fine)
                if np.min(g_check) >= -1e-5:
                    actual_val = compute_value(res.x)
                    if actual_val < best_val:
                        best_val = actual_val
                        best_coeffs = res.x.copy()
                        
                        # Try SLSQP refinement
                        if time.time() < deadline - 15.0:
                            try:
                                res2 = minimize(objective, best_coeffs, method='Nelder-Mead',
                                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-10,
                                                        'adaptive': True})
                                if res2.fun < best_val:
                                    g_check2 = _g_values_fast(res2.x[0], res2.x[1], res2.x[2], x_fine)
                                    if np.min(g_check2) >= -1e-5:
                                        actual_val2 = compute_value(res2.x)
                                        if actual_val2 < best_val:
                                            best_val = actual_val2
                                            best_coeffs = res2.x.copy()
                            except Exception:
                                pass
        except Exception:
            pass
    
    # Final high-precision polish if we found something better
    if best_val < seed_val and time.time() < deadline - 10.0:
        try:
            x_fine = np.linspace(0.001, 5.0, 5000)
            
            def objective_precise(params):
                c0, c1, c2 = params
                val = compute_value([c0, c1, c2])
                if val >= 1e8:
                    return 1e8
                g_vals = _g_values_fast(c0, c1, c2, x_fine)
                penalty = _soft_min_penalty(g_vals, scale=100.0)
                return val + 5.0 * penalty
            
            res_final = minimize(objective_precise, best_coeffs, method='Nelder-Mead',
                                 options={'maxiter': 200, 'xatol': 1e-12, 'fatol': 1e-12,
                                          'adaptive': True})
            g_final = _g_values_fast(res_final.x[0], res_final.x[1], res_final.x[2], x_fine)
            if np.min(g_final) >= -1e-6:
                actual_final = compute_value(res_final.x)
                if actual_final < best_val:
                    best_val = actual_final
                    best_coeffs = res_final.x.copy()
        except Exception:
            pass
    
    if best_val >= seed_val:
        return seed
    
    return best_coeffs
