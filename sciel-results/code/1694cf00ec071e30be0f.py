"""Combined root finder: global Nelder-Mead screening + Newton polish for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0

def _time_left():
    return 150.0 - _elapsed() - _DEADLINE_MARGIN

def _safe_norm(x, F):
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return np.inf
        return float(np.linalg.norm(Fx))
    except Exception:
        return np.inf

def _newton_polish(F, x0, tol=1e-12, max_iter=50):
    x = np.array(x0, dtype=float)
    h = 1e-7
    for _ in range(max_iter):
        if _time_left() < 0.5:
            return x
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= tol:
                return x
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_p = F(x + e)
                Fx_p = np.asarray(Fx_p, dtype=float)
                if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                    return x
                J[:, j] = (Fx_p - Fx) / h
            try:
                dx = np.linalg.solve(J, Fx)
                x_new = x - dx
                if not np.all(np.isfinite(x_new)):
                    return x
                x = x_new
            except np.linalg.LinAlgError:
                return x
        except Exception:
            return x
    return x

def _nelder_mead(F, x0, maxiter=100):
    if _time_left() < 1.0:
        return None
    def objective(x):
        if _time_left() < 0.5:
            return 1e30
        n = _safe_norm(x, F)
        return n * n
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-6, 'fatol': 1e-12})
        if res.success or res.fun < 1e-10:
            return res.x
        return res.x
    except Exception:
        return None

def find_root_system(F, x0):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    x0 = np.asarray(x0, dtype=float).flatten()
    if x0.shape != (2,):
        x0 = np.array([0.0, 0.0])
    
    best_x = None
    best_norm = np.inf

    def _update_best(x_candidate):
        nonlocal best_x, best_norm
        if x_candidate is None:
            return
        try:
            x_candidate = np.asarray(x_candidate, dtype=float).flatten()
            if x_candidate.shape != (2,) or not np.all(np.isfinite(x_candidate)):
                return
            n = _safe_norm(x_candidate, F)
            if n < best_norm:
                best_norm = n
                best_x = x_candidate
        except Exception:
            pass

    # 1. Initial Newton polish from x0
    x_polished = _newton_polish(F, x0)
    _update_best(x_polished)
    if best_norm <= 1e-8:
        return best_x

    # 2. Nelder-Mead global search from x0
    x_nm = _nelder_mead(F, x0)
    if x_nm is not None:
        x_nm_polished = _newton_polish(F, x_nm)
        _update_best(x_nm_polished)
        if best_norm <= 1e-8:
            return best_x

    # 3. Try Nelder-Mead from random points if time permits
    rng = np.random.default_rng(42)
    for i in range(10):
        if _time_left() < 2.0:
            break
        offset = rng.uniform(-10.0, 10.0, size=2)
        x_init = x0 + offset
        x_nm = _nelder_mead(F, x_init, maxiter=50)
        if x_nm is not None:
            x_nm_polished = _newton_polish(F, x_nm)
            _update_best(x_nm_polished)
            if best_norm <= 1e-8:
                return best_x

    # 4. Final high-precision polish if we have a candidate
    if best_x is not None:
        final_x = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
        _update_best(final_x)

    if best_x is not None:
        return best_x
    return x0

# EVOLVE-BLOCK-END
