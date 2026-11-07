import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

# Precompute Hermite polynomial values at x=0
# H0(0) = 1, H4(0) = 12, H8(0) = 1680, H12(0) = 4084080
H0_0 = 1.0
H4_0 = 12.0
H8_0 = 1680.0
H12_0 = 4084080.0

def compute_value(coeffs):
    """Compute the value r^2/(2*pi) where r is the largest sign-changing real root of q(y)=g(sqrt(y))/y."""
    c0, c1, c2 = coeffs
    c3 = -(c0 * H0_0 + c1 * H4_0 + c2 * H8_0) / H12_0
    
    # q(y) = g(sqrt(y))/y = F*y^5 + E*y^4 + D*y^3 + C*y^2 + B*y + A
    # where y = x^2
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
    """Check if g(x) >= 0 for all x in x_grid (excluding x=0 which is 0 by construction)."""
    g_vals = _g_values(coeffs, x_grid)
    # Skip x=0
    return np.min(g_vals[1:]) >= -1e-10

def construct():
    start_time = time.time()
    deadline = start_time + 175.0  # safety margin before 210s kill
    
    seed = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    seed_val = compute_value(seed)
    best_val = seed_val
    best_coeffs = seed.copy()
    
    # Dense grid for positivity check
    x_max_grid = 20.0
    n_grid = 3000
    x_grid = np.linspace(0.0, x_max_grid, n_grid)
    
    beta = 200.0  # Sharpness of softplus penalty
    
    def objective(params):
        """Smooth penalty-based objective: minimize r^2/(2*pi) subject to g(x) >= 0."""
        c0, c1, c2 = params
        coeffs = [c0, c1, c2]
        
        # Compute the actual value
        val = compute_value(coeffs)
        
        # Smooth positivity penalty
        g_vals = _g_values(coeffs, x_grid)
        t = -beta * g_vals
        t_clipped = np.clip(t, -50, 50)
        penalty = np.sum(np.log1p(np.exp(t_clipped)))
        
        return val + penalty
    
    # Multi-start Nelder-Mead optimization with adaptive strategy
    rng = np.random.default_rng(42)
    
    # Create a diverse set of initial points
    start_points = [seed.copy()]
    
    # Points near seed with varying scales
    for scale in [0.1, 0.5, 1.0, 2.0]:
        for _ in range(10):
            if time.time() > deadline - 30.0:
                break
            start_points.append(seed + rng.normal(0, scale, size=3))
    
    # Random points in a reasonable range
    for _ in range(50):
        if time.time() > deadline - 30.0:
            break
        start_points.append(rng.uniform(-3.0, 3.0, size=3))
    
    # Points biased toward making c0 more negative (to potentially reduce r)
    for _ in range(20):
        if time.time() > deadline - 30.0:
            break
        c0 = rng.uniform(-2.0, -0.5)
        c1 = rng.uniform(-0.5, 0.5)
        c2 = rng.uniform(-0.1, 0.1)
        start_points.append(np.array([c0, c1, c2]))
    
    for start in start_points:
        if time.time() > deadline - 15.0:
            break
        
        try:
            # Phase 1: Coarse optimization
            res = minimize(objective, start, method='Nelder-Mead',
                           options={'maxiter': 300, 'xatol': 1e-6, 'fatol': 1e-6,
                                    'adaptive': True})
            
            if res.fun < best_val * 1.01:  # Slightly better than best
                # Verify positivity
                if _check_positivity(res.x, x_grid):
                    # Phase 2: Fine polish
                    res_polish = minimize(objective, res.x, method='Nelder-Mead',
                                          options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10,
                                                   'adaptive': True})
                    if res_polish.fun < best_val and _check_positivity(res_polish.x, x_grid):
                        best_val = res_polish.fun
                        best_coeffs = res_polish.x.copy()
                        
                        # Phase 3: Very tight local search around best point
                        if time.time() < deadline - 20.0:
                            for _ in range(3):
                                if time.time() > deadline - 15.0:
                                    break
                                res_tight = minimize(objective, best_coeffs + rng.normal(0, 1e-5, size=3),
                                                     method='Nelder-Mead',
                                                     options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-12,
                                                              'adaptive': True})
                                if res_tight.fun < best_val and _check_positivity(res_tight.x, x_grid):
                                    best_val = res_tight.fun
                                    best_coeffs = res_tight.x.copy()
        except Exception:
            pass
    
    # Final verification and return
    if best_val >= seed_val:
        return seed
    
    return best_coeffs
