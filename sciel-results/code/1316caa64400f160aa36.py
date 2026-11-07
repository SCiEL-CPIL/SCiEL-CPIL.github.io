"""Combined root finder: Broyden's method with damped Newton fallback for 3 fixed 2D systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, fsolve

_START_TIME = None
_DEADLINE_MARGIN = 10  # seconds to spare before 150s limit

def _elapsed():
    return time.time() - _START_TIME if _START_TIME is not None else 0.0

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

def _broyden(F, x0, tol=1e-12, max_iter=50):
    """
    Broyden's method (good direction) with damping.
    x_{k+1} = x_k - alpha * J_k^{-1} F(x_k)
    J_{k+1} = J_k + (s_k - J_k y_k) s_k^T / (s_k^T s_k)
    where s_k = x_{k+1} - x_k, y_k = F(x_{k+1}) - F(x_k)
    """
    x = np.array(x0, dtype=float)
    h = 1e-7
    # Initial Jacobian via finite differences
    J = np.zeros((2, 2))
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            return x
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = F(x + e)
            Fx_p = np.asarray(Fx_p, dtype=float)
            if Fx_p.shape != (2,) or not np.all(np.isfinite(Fx_p)):
                return x
            J[:, j] = (Fx_p - Fx) / h
    except Exception:
        return x

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
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                return x
            
            # Damping: start with alpha=1, reduce if it doesn't decrease norm
            alpha = 1.0
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            Fx_new = np.asarray(Fx_new, dtype=float)
            if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                return x
            norm_Fx_new = np.linalg.norm(Fx_new)
            
            # Backtracking line search
            while norm_Fx_new >= norm_Fx and alpha > 1e-6:
                alpha *= 0.5
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                Fx_new = np.asarray(Fx_new, dtype=float)
                if Fx_new.shape != (2,) or not np.all(np.isfinite(Fx_new)):
                    return x
                norm_Fx_new = np.linalg.norm(Fx_new)
            
            if alpha < 1e-6:
                return x
                
            s = x_new - x
            y = Fx_new - Fx
            s_norm_sq = np.dot(s, s)
            if s_norm_sq < 1e-15:
                return x
                
            # Broyden update
            J = J + np.outer(s - J @ y, s) / s_norm_sq
            
            x = x_new
            Fx = Fx_new
            norm_Fx = norm_Fx_new
            
        except Exception:
            return x
    return x

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

    # 1. Initial Broyden from x0
    x_broyden = _broyden(F, x0)
    _update_best(x_broyden)
    if best_norm <= 1e-8:
        # Final polish to ensure high precision
        x_final = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
        _update_best(x_final)
        return best_x

    # 2. If Broyden failed or didn't converge well, try Newton polish from x0
    x_polished = _newton_polish(F, x0)
    _update_best(x_polished)
    if best_norm <= 1e-8:
        x_final = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
        _update_best(x_final)
        return best_x

    # 3. Try Broyden from slightly perturbed points if time permits
    rng = np.random.default_rng(42)
    for i in range(20):
        if _time_left() < 2.0:
            break
        # Small perturbations to escape local minima or bad initial Jacobian estimates
        offset = rng.uniform(-1.0, 1.0, size=2)
        x_init = x0 + offset
        x_broyden = _broyden(F, x_init, max_iter=30)
        _update_best(x_broyden)
        if best_norm <= 1e-8:
            x_final = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
            _update_best(x_final)
            return best_x

    # 4. Fallback to SciPy's hybr (Levenberg-Marquardt) if still failing
    if _time_left() > 5.0:
        try:
            res = fsolve(F, x0, full_output=True)
            x_sol = res[0]
            _update_best(x_sol)
            if best_norm <= 1e-8:
                x_final = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
                _update_best(x_final)
                return best_x
        except Exception:
            pass

    # 5. Final high-precision polish if we have a candidate
    if best_x is not None:
        final_x = _newton_polish(F, best_x, tol=1e-14, max_iter=20)
        _update_best(final_x)

    if best_x is not None:
        return best_x
    return x0

# EVOLVE-BLOCK-END
