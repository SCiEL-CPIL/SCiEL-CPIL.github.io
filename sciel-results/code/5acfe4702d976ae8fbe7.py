import numpy as np
from scipy import optimize
import time

def compute_value(c0, c1, c2):
    H0_0 = 1.0
    H4_0 = 12.0
    H8_0 = 1680.0
    H12_0 = 1032160.0
    c3 = -(c0*H0_0 + c1*H4_0 + c2*H8_0) / H12_0
    
    coeffs = np.array([
        4096.0*c3,
        -98304.0*c3,
        846720.0*c3 + 256.0*c2,
        -3225600.0*c3 + 11520.0*c2,
        4976640.0*c3 - 11520.0*c2 + 16.0*c1,
        -2419200.0*c3 - 11520.0*c2 - 48.0*c1
    ])
    
    # Check if leading coefficient is zero (degenerate)
    if abs(coeffs[0]) < 1e-15:
        # Reduce degree
        while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
            coeffs = coeffs[1:]
    
    roots = np.roots(coeffs)
    real_roots = roots[np.abs(roots.imag) < 1e-9].real
    real_roots = real_roots[real_roots > 1e-9]
    
    if len(real_roots) == 0:
        return np.inf
    
    changing_roots = []
    for t0 in real_roots:
        eps = 1e-7 * max(1.0, abs(t0))
        f_minus = np.polyval(coeffs, t0 - eps)
        f_plus = np.polyval(coeffs, t0 + eps)
        if f_minus * f_plus < 0:
            changing_roots.append(t0)
    
    if len(changing_roots) == 0:
        return np.inf
    
    t_max = max(changing_roots)
    return t_max / (2.0 * np.pi)

def construct():
    start_time = time.time()
    deadline = start_time + 175.0
    
    seed_params = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_value = compute_value(*seed_params)
    best_params = seed_params.copy()
    
    rng = np.random.default_rng(123)
    
    # Phase 1: Differential evolution for global search
    bounds = [(-5, 5), (-1, 1), (-0.01, 0.01)]
    
    def obj(x):
        return compute_value(x[0], x[1], x[2])
    
    # Try DE with limited time
    try:
        de_result = optimize.differential_evolution(
            obj, bounds, seed=123, maxiter=50, popsize=20, tol=1e-8,
            polish=False, init='sobol'
        )
        if time.time() < deadline - 60:
            val = obj(de_result.x)
            if val < best_value:
                best_value = val
                best_params = de_result.x.copy()
    except:
        pass
    
    # Phase 2: Local refinement with multiple restarts
    n_restarts = 50
    for i in range(n_restarts):
        if time.time() > deadline - 20:
            break
        
        # Start near best or random
        if i < 10:
            x0 = best_params + rng.normal(0, [0.1, 0.01, 0.001], 3)
        else:
            x0 = rng.uniform(-2, 2, 3)
            x0[2] *= 0.01
        
        try:
            result = optimize.minimize(
                obj, x0, method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            val = obj(result.x)
            if val < best_value:
                best_value = val
                best_params = result.x.copy()
        except:
            pass
    
    return np.array(best_params)
