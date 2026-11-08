import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import minimize, differential_evolution
from numpy.polynomial.laguerre import laggauss
import time

_START_TIME = time.time()
_TIMEOUT = 600.0
_SAFETY = 30.0

def _time_left():
    return _TIMEOUT - _SAFETY - (time.time() - _START_TIME)

def _compute_c4(roots):
    """
    Compute C4 upper bound for 12 positive double roots.
    
    P(x) = prod_{i=1}^{12} (x - r_i)^2  (degree 24)
    
    Express P in Laguerre basis: P = sum a_k L_k^(1)(x)
    C4 = sum_{k=0}^{11} a_k^2 / ((k+1)(k+2))
    """
    # Build polynomial coefficients (ascending order)
    coeff = np.array([1.0])
    for r in roots:
        # multiply by (x - r)^2 = x^2 - 2r x + r^2
        new_c = np.zeros(len(coeff) + 2)
        new_c[:len(coeff)] += coeff * (r * r)
        new_c[1:len(coeff)+1] += coeff * (-2 * r)
        new_c[2:len(coeff)+2] += coeff
        coeff = new_c
    
    # Gauss-Laguerre quadrature with weight e^{-x}
    nodes, weights = laggauss(80)
    P_vals = np.polyval(coeff[::-1], nodes)
    
    # Compute Laguerre coefficients a_k for k=0..11
    c4 = 0.0
    for k in range(12):
        Lk = eval_genlaguerre(k, 1, nodes)
        a_k = np.sum(weights * P_vals * Lk) / ((k + 1) * (k + 2))
        c4 += a_k * a_k / ((k + 1) * (k + 2))
    
    return c4

def _objective(roots):
    if np.any(roots <= 0) or np.any(np.diff(np.sort(roots)) < 1e-8):
        return 1e10
    try:
        v = _compute_c4(roots)
        return v if np.isfinite(v) else 1e10
    except:
        return 1e10

def construct():
    best_roots = None
    best_score = np.inf
    
    # Starting points
    nodes, _ = laggauss(12)
    starts = [nodes + 1.0]
    starts.append(np.linspace(0.5, 20, 12))
    starts.append(np.sort(np.random.uniform(0.5, 30, 12)))
    
    # DE for global search
    if _time_left() > 150:
        try:
            res = differential_evolution(_objective, [(0.1, 50.0)] * 12,
                                         maxiter=100, popsize=12, seed=42, polish=True)
            if res.fun < best_score:
                best_score, best_roots = res.fun, res.x.copy()
        except:
            pass
    
    # Local polish
    for s in starts:
        if _time_left() < 20:
            break
        try:
            r = minimize(_objective, s, method='L-BFGS-B',
                         bounds=[(0.1, 50.0)] * 12, options={'maxiter': 300})
            if r.fun < best_score:
                best_score, best_roots = r.fun, r.x.copy()
        except:
            pass
    
    if best_roots is None:
        return list(starts[0])
    
    return list(np.sort(np.abs(best_roots)) + 1e-8)
