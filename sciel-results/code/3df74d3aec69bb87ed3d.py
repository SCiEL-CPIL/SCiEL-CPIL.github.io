import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

def _hermite(x, n):
    """Physicists' Hermite polynomial H_n(x) for n in {0, 4, 8, 12}."""
    if n == 0:
        return np.ones_like(x, dtype=np.float64)
    elif n == 4:
        return 16.0*x**4 - 48.0*x**2 + 12.0
    elif n == 8:
        return 256.0*x**8 - 4608.0*x**6 + 26880.0*x**4 - 43008.0*x**2 + 8064.0
    elif n == 12:
        return (4096.0*x**12 - 706560.0*x**10 + 16515072.0*x**8 
                - 123552000.0*x**6 + 315302400.0*x**4 - 275187840.0*x**2 + 47900160.0)
    else:
        raise ValueError(f"Hermite order {n} not supported")

def _g(x, c0, c4, c8):
    """g(x) = (c0 H0 + c4 H4 + c8 H8 + c12 H12) exp(-x^2/2), with c12 from g(0)=0."""
    H0 = np.ones_like(x, dtype=np.float64)
    H4 = _hermite(x, 4)
    H8 = _hermite(x, 8)
    H12 = _hermite(x, 12)
    # H0(0)=1, H4(0)=12, H8(0)=8064, H12(0)=47900160
    c12 = -(c0 + 12.0*c4 + 8064.0*c8) / 47900160.0
    poly = c0*H0 + c4*H4 + c8*H8 + c12*H12
    return poly * np.exp(-0.5*x**2)

def _obj_linf(c):
    c0, c4, c8 = c
    x = np.linspace(-8.0, 8.0, 4096)
    return np.max(np.abs(_g(x, c0, c4, c8)))

def _obj_l1(c):
    c0, c4, c8 = c
    x = np.linspace(-8.0, 8.0, 4096)
    g = _g(x, c0, c4, c8)
    return np.trapz(np.abs(g), x)

def construct():
    t0 = time.time()
    DEADLINE = 170.0  # leave margin
    
    # Baseline
    c_base = np.array([-113.0/100.0, 1.0/25.0, 1.0/3240.0])
    best_c = c_base.copy()
    best_val = _obj_linf(best_c)
    
    # Local refinement with Nelder-Mead
    try:
        res = minimize(_obj_linf, best_c, method='Nelder-Mead',
                       options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True})
        if time.time() - t0 > DEADLINE - 40:
            return best_c
        if res.fun < best_val:
            best_val = res.fun
            best_c = res.x.copy()
    except Exception:
        pass
    
    # Global search with differential evolution
    if time.time() - t0 < DEADLINE - 60:
        try:
            bounds = [(-3.0, 3.0), (-0.5, 0.5), (-0.005, 0.005)]
            res2 = differential_evolution(_obj_linf, bounds, maxiter=30, popsize=20,
                                           seed=12345, tol=1e-10, polish=True)
            if time.time() - t0 > DEADLINE - 20:
                return best_c
            if res2.fun < best_val:
                best_val = res2.fun
                best_c = res2.x.copy()
        except Exception:
            pass
    
    return best_c
