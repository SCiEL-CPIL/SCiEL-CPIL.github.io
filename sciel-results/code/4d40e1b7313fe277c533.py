"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 400

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize_scalar

def _value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = np.convolve(h, h)
    return abs(2 * N * np.max(conv) / (s ** 2))

def _objective(alpha):
    h = 1.0 + alpha * np.cos(2 * np.pi * np.arange(N) / N)
    return _value(h)

def construct():
    deadline = time.time() + 100  # safety margin before 120s limit
    
    # Start with seed
    best_h = np.ones(N)
    best_val = _value(best_h)  # 2.0
    
    # Try sinusoidal perturbation: h[i] = 1 + alpha * cos(2*pi*i/N)
    # Optimize alpha in a reasonable range
    try:
        res = minimize_scalar(_objective, bounds=(-2.0, 2.0), method='bounded',
                             options={'xatol': 1e-10, 'maxiter': 200})
        if res.success and res.fun < best_val:
            alpha_opt = res.x
            best_h = 1.0 + alpha_opt * np.cos(2 * np.pi * np.arange(N) / N)
            best_val = res.fun
    except Exception:
        pass
    
    # Try different frequencies: h[i] = 1 + alpha * cos(2*pi*k*i/N)
    for k in [1, 2, 3, 5, 7, 10, 13, 17, 20, 25, 33, 50, 100, 200]:
        if time.time() > deadline:
            break
        try:
            def obj_k(alpha):
                h = 1.0 + alpha * np.cos(2 * np.pi * k * np.arange(N) / N)
                return _value(h)
            
            res = minimize_scalar(obj_k, bounds=(-2.0, 2.0), method='bounded',
                                 options={'xatol': 1e-10, 'maxiter': 100})
            if res.success and res.fun < best_val:
                alpha_opt = res.x
                best_h = 1.0 + alpha_opt * np.cos(2 * np.pi * k * np.arange(N) / N)
                best_val = res.fun
        except Exception:
            pass
    
    # Try linear combination of two frequencies
    if time.time() < deadline:
        for k1, k2 in [(1, 2), (1, 3), (2, 3), (1, 5), (2, 5), (3, 5), (1, 7), (2, 7), (3, 7)]:
            if time.time() > deadline:
                break
            try:
                def obj_2d(alpha):
                    a1, a2 = alpha
                    h = 1.0 + a1 * np.cos(2 * np.pi * k1 * np.arange(N) / N) + \
                          a2 * np.cos(2 * np.pi * k2 * np.arange(N) / N)
                    return _value(h)
                
                from scipy.optimize import minimize
                x0 = np.array([0.5, 0.5])
                res = minimize(obj_2d, x0, method='Nelder-Mead',
                             options={'xatol': 1e-8, 'maxiter': 200, 'adaptive': True})
                if res.success and res.fun < best_val:
                    a1, a2 = res.x
                    best_h = 1.0 + a1 * np.cos(2 * np.pi * k1 * np.arange(N) / N) + \
                              a2 * np.cos(2 * np.pi * k2 * np.arange(N) / N)
                    best_val = res.fun
            except Exception:
                pass
    
    # Final polish: local optimization around best solution
    if time.time() < deadline - 10:
        try:
            from scipy.optimize import minimize
            
            # Use gradient-free local optimization
            def obj_full(h_flat):
                h = h_flat.reshape(N)
                return _value(h)
            
            res = minimize(obj_full, best_h.flatten(), method='Nelder-Mead',
                         options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 500, 'adaptive': True})
            if res.success and res.fun < best_val:
                best_h = res.x.reshape(N)
                best_val = res.fun
        except Exception:
            pass
    
    return best_h
# EVOLVE-BLOCK-END
