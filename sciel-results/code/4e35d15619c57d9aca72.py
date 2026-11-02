# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad

_SEED = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])

# Hermite polynomial values at 0
_H0_0 = 1.0
_H4_0 = 12.0
_H8_0 = 1680.0
_H12_0 = 665280.0

def _get_poly_coeffs(a0, a4, a8):
    a12 = -(a0 * _H0_0 + a4 * _H4_0 + a8 * _H8_0) / _H12_0
    c12 = 4096.0 * a12
    c10 = -24576.0 * a12 + 256.0 * a8
    c8 = 430080.0 * a12 - 3072.0 * a8 + 16.0 * a4
    c6 = -2457600.0 * a12 + 10752.0 * a8 - 48.0 * a4
    c4 = 6220800.0 * a12 - 10752.0 * a8 + 12.0 * a4 + a0
    c2 = -5443200.0 * a12 + 1680.0 * a8 - 48.0 * a4
    c0 = 665280.0 * a12 + 1680.0 * a8 + 12.0 * a4 + a0
    
    coeffs = np.zeros(13)
    coeffs[0] = c12
    coeffs[2] = c10
    coeffs[4] = c8
    coeffs[6] = c6
    coeffs[8] = c4
    coeffs[10] = c2
    coeffs[12] = c0
    return coeffs, a12, (c12, c10, c8, c6, c4, c2, c0)

def _objective(c):
    a0, a4, a8 = c
    coeffs, a12, (c12, c10, c8, c6, c4, c2, c0) = _get_poly_coeffs(a0, a4, a8)
    
    try:
        roots = np.roots(coeffs)
    except Exception:
        return 1e30
        
    real_roots = []
    for r in roots:
        if abs(r.imag) < 1e-10:
            r_real = float(r.real)
            if -10.0 <= r_real <= 10.0:
                real_roots.append(r_real)
                
    real_roots.sort()
    
    def integrand(x):
        y = x * x
        p_val = c12
        p_val = p_val * y + c10
        p_val = p_val * y + c8
        p_val = p_val * y + c6
        p_val = p_val * y + c4
        p_val = p_val * y + c2
        p_val = p_val * y + c0
        
        g_val = p_val * np.exp(-0.5 * y)
        return abs(g_val)
        
    if not real_roots:
        try:
            L1, err = quad(integrand, -10.0, 10.0, limit=200, epsabs=1e-12, epsrel=1e-12)
        except Exception:
            return 1e30
    else:
        points = real_roots
        try:
            L1, err = quad(integrand, -10.0, 10.0, points=points, limit=300, epsabs=1e-12, epsrel=1e-12)
        except Exception:
            try:
                L1, err = quad(integrand, -10.0, 10.0, limit=300, epsabs=1e-12, epsrel=1e-12)
            except Exception:
                return 1e30
                
    if not np.isfinite(L1) or L1 < 0:
        return 1e30
        
    return L1 * L1

def construct():
    t0 = time.time()
    deadline = t0 + 200.0  # Safety margin before 210s kill
    
    best_c = _SEED.copy()
    best_val = _objective(best_c)
    
    # Phase 1: Global search with differential evolution
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)]
    
    if time.time() < deadline - 80.0:
        try:
            res_global = differential_evolution(
                _objective, bounds, seed=42, maxiter=200, popsize=20, 
                tol=1e-10, polish=False, mutation=(0.5, 1.5), recombination=0.8
            )
            if res_global.fun < best_val:
                best_val = float(res_global.fun)
                best_c = np.asarray(res_global.x, dtype=float).copy()
        except Exception:
            pass
    
    # Phase 2: Multiple local polish attempts from different starting points
    start_points = [best_c, _SEED.copy()]
    # Add some perturbed starts
    rng = np.random.RandomState(123)
    for _ in range(3):
        start_points.append(_SEED + rng.normal(0, 0.5, 3))
    
    for start in start_points:
        if time.time() >= deadline - 30.0:
            break
        try:
            res1 = minimize(_objective, start, method='L-BFGS-B', 
                           options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-16})
            if res1.fun < best_val:
                best_val = float(res1.fun)
                best_c = np.asarray(res1.x, dtype=float).copy()
        except Exception:
            pass
    
    # Phase 3: High-precision polish with Nelder-Mead
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_objective, best_c, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-18, 'fatol': 1e-18, 'adaptive': True})
            if res2.fun < best_val:
                best_val = float(res2.fun)
                best_c = np.asarray(res2.x, dtype=float).copy()
        except Exception:
            pass
    
    # Phase 4: Final ultra-precise BFGS polish
    if time.time() < deadline - 2.0:
        try:
            res3 = minimize(_objective, best_c, method='BFGS',
                           options={'maxiter': 5000, 'gtol': 1e-20, 'disp': False})
            if res3.fun < best_val:
                best_val = float(res3.fun)
                best_c = np.asarray(res3.x, dtype=float).copy()
        except Exception:
            pass
    
    if not np.all(np.isfinite(best_c)):
        return _SEED.copy()
        
    return best_c
# EVOLVE-BLOCK-END
