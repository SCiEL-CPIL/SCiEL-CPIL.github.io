import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

# Precompute Hermite polynomial values at x=0
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4084080.0

# q(y) = g(sqrt(y))/y has coefficients:
# q(y) = F*y^5 + E*y^4 + D*y^3 + C*y^2 + B*y + A
# where:
# A = -48*c1 - 13440*c2 - 11827200*c3
# B = 16*c1 + 11520*c2 + 13977600*c3
# C = -3072*c2 - 6812160*c3
# D = 256*c2 + 1397760*c3
# E = -122880*c3
# F = 4096*c3
# And c3 = -(c0 + 12*c1 + 1680*c2) / 4084080

def compute_value(coeffs):
    """Compute r^2/(2*pi) where r is the largest sign-changing positive root of q(y)."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
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

def _g_values(coeffs, x_grid):
    """Evaluate g(x) at points in x_grid."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
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

def _check_positivity(coeffs, x_grid):
    """Check if g(x) >= 0 for all x in x_grid (excluding x=0)."""
    g_vals = _g_values(coeffs, x_grid)
    return np.min(g_vals[1:]) >= -1e-10

def _soft_penalty(g_vals, beta=200.0):
    """Smooth softplus penalty for negative values."""
    t = -beta * g_vals
    t_clipped = np.clip(t, -50, 50)
    return np.sum(np.log1p(np.exp(t_clipped)))

def _objective(params, x_grid, beta=200.0):
    """Objective: minimize r^2/(2*pi) + soft penalty for g(x) < 0."""
    c0, c1, c2 = params
    val = compute_value([c0, c1, c2])
    g_vals = _g_values([c0, c1, c2], x_grid)
    penalty = _soft_penalty(g_vals, beta)
    return val + penalty

def _objective_grad(params, x_grid, beta=200.0):
    """Objective with numerical gradient for BFGS."""
    val = compute_value(params)
    g_vals = _g_values(params, x_grid)
    penalty = _soft_penalty(g_vals, beta)
    return val + penalty

def construct():
    start_time = time.time()
    deadline = start_time + 170.0  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Dense grid for positivity check - use log-spaced grid for better coverage
    x_grid = np.concatenate([
        np.linspace(0.0, 5.0, 2000),
        np.linspace(5.0, 20.0, 1000)
    ])
    
    beta = 200.0
    
    rng = np.random.default_rng(42)
    
    # Strategy: Multi-phase optimization
    # Phase 1: Coarse search with Nelder-Mead from multiple starts
    # Phase 2: Refine with BFGS using numerical gradients
    # Phase 3: Very tight local polish
    
    # Generate diverse starting points
    start_points = [seed.copy()]
    
    # Points near seed
    for scale in [0.05, 0.1, 0.3, 0.5, 1.0]:
        for _ in range(5):
            start_points.append(seed + rng.normal(0, scale, size=3))
    
    # Points with more negative c0
    for _ in range(20):
        c0 = rng.uniform(-2.5, -0.8)
        c1 = rng.uniform(-0.3, 0.3)
        c2 = rng.uniform(-0.05, 0.05)
        start_points.append(np.array([c0, c1, c2]))
    
    # Random points
    for _ in range(30):
        start_points.append(rng.uniform(-2.0, 2.0, size=3))
    
    # Points specifically designed to reduce r
    # Try to make q(y) have smaller positive roots
    for _ in range(15):
        c0 = rng.uniform(-1.5, -0.5)
        c1 = rng.uniform(0.01, 0.1)
        c2 = rng.uniform(-0.02, 0.02)
        start_points.append(np.array([c0, c1, c2]))
    
    for i, start in enumerate(start_points):
        if time.time() > deadline - 20.0:
            break
        
        try:
            # Phase 1: Coarse Nelder-Mead
            res = minimize(_objective, start, args=(x_grid, beta), method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5,
                                    'adaptive': True})
            
            if res.fun < best_val * 1.02:
                # Phase 2: BFGS refinement
                if time.time() < deadline - 15.0:
                    res_bfgs = minimize(_objective, res.x, args=(x_grid, beta), method='BFGS',
                                       options={'maxiter': 100, 'gtol': 1e-6})
                    if res_bfgs.fun < best_val:
                        # Verify positivity
                        if _check_positivity(res_bfgs.x, x_grid):
                            best_val = res_bfgs.fun
                            best_coeffs = res_bfgs.x.copy()
                            
                            # Phase 3: Tight polish with very fine tolerances
                            if time.time() < deadline - 10.0:
                                for _ in range(3):
                                    if time.time() > deadline - 8.0:
                                        break
                                    res_tight = minimize(_objective, best_coeffs + rng.normal(0, 1e-6, size=3),
                                                         args=(x_grid, beta), method='BFGS',
                                                         options={'maxiter': 200, 'gtol': 1e-8})
                                    if res_tight.fun < best_val and _check_positivity(res_tight.x, x_grid):
                                        best_val = res_tight.fun
                                        best_coeffs = res_tight.x.copy()
        except Exception:
            pass
    
    # Final check: if we haven't improved, return seed
    if best_val >= seed_val:
        return seed
    
    return best_coeffs
