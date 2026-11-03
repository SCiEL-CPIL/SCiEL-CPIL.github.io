"""Refined Cohn-Goncalves Laguerre construction: geometric-root + direct optimization."""
import time
import math
import numpy as np
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 600.0
_MARGIN = 30.0


def _time_left():
    return _DEADLINE - (time.time() - _START) - _MARGIN


def _laguerre_coeffs(n, alpha=1):
    """Compute coefficients of L_n^(alpha)(x) in ascending power of x."""
    coeff = np.zeros(n + 1)
    for k in range(n + 1):
        try:
            log_val = math.lgamma(n + alpha + 1) - math.lgamma(n - k + 1) - math.lgamma(k + alpha + 1)
            binom_val = math.exp(log_val)
        except Exception:
            binom_val = 1.0
        coeff[k] = ((-1) ** k) * binom_val / math.factorial(k)
    return coeff


def _compute_integral(poly):
    """Compute integral_0^inf psi(x)^2 x^1 e^{-x} dx for psi with ascending coeffs."""
    sq = np.convolve(poly, poly)
    max_k = len(sq) - 1
    weights = np.ones(max_k + 1)
    for j in range(1, max_k + 1):
        weights[j] = weights[j - 1] * (j + 1)
    return np.dot(sq, weights)


def _c4_bound(roots, m=12):
    """Upper bound on C4 = 1 / integral."""
    if _time_left() < 5:
        return float('inf')
    
    coeff_L = _laguerre_coeffs(m, 1)
    poly = coeff_L.copy()
    for r in roots:
        if r <= 0:
            return float('inf')
        q = np.array([r * r, -2.0 * r, 1.0])
        poly = np.convolve(poly, q)
    
    if not np.all(np.isfinite(poly)):
        return float('inf')
    
    val = _compute_integral(poly)
    if not np.isfinite(val) or val <= 0:
        return float('inf')
    return 1.0 / val


def _objective_geometric(log_params):
    """Objective for geometric roots: minimize C4 bound."""
    log_r0, log_q = log_params
    r0 = np.exp(log_r0)
    q = np.exp(log_q)
    
    if r0 <= 0 or q <= 0:
        return 1e10
    
    roots = r0 * (q ** np.arange(12))
    
    if np.any(roots <= 0) or np.any(roots > 1e10) or np.any(roots < 1e-10):
        return 1e10
    
    return _c4_bound(roots)


def _nelder_mead_minimize(func, x0, bounds_low, bounds_high, max_iter=200):
    """Simple Nelder-Mead to minimize func. Returns best x and value."""
    n = len(x0)
    simplex = [np.array(x0, dtype=float)]
    for i in range(n):
        x = np.array(x0, dtype=float)
        step = 0.1 * (bounds_high[i] - bounds_low[i]) if bounds_high[i] > bounds_low[i] else 1.0
        x[i] += step
        x = np.clip(x, bounds_low, bounds_high)
        simplex.append(x)
    
    func_vals = [func(x) for x in simplex]
    best_x = x0[:]
    best_val = func(best_x)
    
    for it in range(max_iter):
        if _time_left() < 30:
            break
        order = np.argsort(func_vals)
        simplex = [simplex[i] for i in order]
        func_vals = [func_vals[i] for i in order]
        
        if func_vals[0] < best_val:
            best_val = func_vals[0]
            best_x = simplex[0].copy()
        
        centroid = np.mean(simplex[:-1], axis=0)
        
        xr = centroid + (centroid - simplex[-1])
        xr = np.clip(xr, bounds_low, bounds_high)
        fr = func(xr)
        
        if fr <= func_vals[0]:
            xe = centroid + 2.0 * (centroid - simplex[-1])
            xe = np.clip(xe, bounds_low, bounds_high)
            fe = func(xe)
            if fe < fr:
                simplex[-1] = xe
                func_vals[-1] = fe
            else:
                simplex[-1] = xr
                func_vals[-1] = fr
        elif fr < func_vals[-2]:
            simplex[-1] = xr
            func_vals[-1] = fr
        else:
            xc = centroid + 0.5 * (simplex[-1] - centroid)
            xc = np.clip(xc, bounds_low, bounds_high)
            fc = func(xc)
            if fc < func_vals[-1]:
                simplex[-1] = xc
                func_vals[-1] = fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                    simplex[i] = np.clip(simplex[i], bounds_low, bounds_high)
                    func_vals[i] = func(simplex[i])
    
    return best_x, best_val


def construct():
    m = 12
    n_roots = 12
    
    seed_roots = [4.0 * (i + 1) for i in range(n_roots)]
    seed_val = _c4_bound(seed_roots, m)
    
    best_roots = seed_roots[:]
    best_val = seed_val
    
    # Phase 1: Optimize in geometric parameter space (log r0, log q)
    if _time_left() > 120:
        np.random.seed(42)
        
        # Fit geometric to seed: 4, 8, ..., 48
        # log(4), log(8), ..., log(48) -> linear fit gives log_r0, log_q
        seed_logs = np.log(seed_roots)
        x = np.arange(n_roots)
        A = np.vstack([np.ones_like(x), x]).T
        log_r0_fit, log_q_fit = np.linalg.lstsq(A, seed_logs, rcond=None)[0]
        
        start_points = [
            (log_r0_fit, log_q_fit),
            (math.log(4.0), math.log(12 ** (1.0 / 11))),
            (math.log(2.0), math.log(2.0)),
            (math.log(1.0), math.log(1.5)),
            (math.log(5.0), math.log(1.3)),
            (math.log(3.0), math.log(1.8)),
            (math.log(1.0), math.log(2.0)),
            (math.log(10.0), math.log(1.2)),
            (math.log(0.5), math.log(2.5)),
        ]
        
        for _ in range(8):
            lr0 = np.random.uniform(math.log(0.1), math.log(20.0))
            lq = np.random.uniform(math.log(0.5), math.log(4.0))
            start_points.append((lr0, lq))
        
        for sp in start_points:
            if _time_left() < 60:
                break
            
            try:
                result = minimize(
                    _objective_geometric,
                    x0=np.array(sp),
                    method='L-BFGS-B',
                    bounds=[(math.log(0.01), math.log(100.0)), (math.log(0.1), math.log(10.0))],
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                
                if result.fun < best_val:
                    log_r0, log_q = result.x
                    r0 = np.exp(log_r0)
                    q = np.exp(log_q)
                    roots = r0 * (q ** np.arange(n_roots))
                    
                    if np.all(roots > 0) and np.all(np.isfinite(roots)):
                        score = _c4_bound(roots, m)
                        if score < best_val:
                            best_val = score
                            best_roots = [float(r) for r in roots]
            except Exception:
                continue
    
    # Phase 2: Direct optimization of all 12 roots
    if _time_left() > 60:
        def objective(roots_arr):
            if np.any(roots_arr <= 0):
                return float('inf')
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_low = np.ones(n_roots) * 0.1
        bounds_high = np.ones(n_roots) * 100.0
        
        best_x, best_f = _nelder_mead_minimize(
            objective, x0, bounds_low, bounds_high, max_iter=300
        )
        
        if np.isfinite(best_f) and best_f < best_val and np.all(best_x > 0):
            best_val = best_f
            best_roots = list(best_x)
    
    # Phase 3: Fine-tune with perturbations
    if _time_left() > 30 and best_val < seed_val:
        def objective2(roots_arr):
            if np.any(roots_arr <= 0):
                return float('inf')
            return _c4_bound(roots_arr, m)
        
        x0 = np.array(best_roots, dtype=float)
        bounds_low = np.ones(n_roots) * 0.05
        bounds_high = np.ones(n_roots) * 150.0
        
        for restart in range(5):
            if _time_left() < 20:
                break
            x0_pert = x0 + np.random.normal(0, 0.1 * np.max(x0), n_roots)
            x0_pert = np.clip(x0_pert, bounds_low, bounds_high)
            
            best_x, best_f = _nelder_mead_minimize(
                objective2, x0_pert, bounds_low, bounds_high, max_iter=100
            )
            
            if np.isfinite(best_f) and best_f < best_val and np.all(best_x > 0):
                best_val = best_f
                best_roots = list(best_x)
                x0 = best_x
    
    return best_roots
