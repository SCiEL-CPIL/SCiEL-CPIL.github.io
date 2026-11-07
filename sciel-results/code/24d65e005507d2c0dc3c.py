"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
import numpy as np
import time

_DEADLINE_FRAC = 0.85
_CALL_LIMIT = 490


def _elapsed_ok(start, limit_frac=0.85, wall=120.0):
    return (time.time() - start) < limit_frac * wall


def _finite2(x):
    if not isinstance(x, np.ndarray):
        return False
    if x.shape != (2,):
        return False
    return bool(np.all(np.isfinite(x)))


def _norm2(v):
    return float(np.linalg.norm(v))


def find_root_system(F, x0):
    start = time.time()
    x = np.array(x0, dtype=float)
    
    # Initial evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
    except Exception:
        return x
    
    if not _finite2(Fx):
        return x
    
    best_x = x.copy()
    best_norm = _norm2(Fx)
    calls = 1
    
    if best_norm <= 1e-8:
        return best_x
    
    # Initialize Broyden Jacobian estimate
    # Use finite difference for initial Jacobian (2 extra calls)
    h = 1e-5
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_p = np.asarray(F(x + e), dtype=float)
            calls += 1
            if _finite2(Fx_p):
                J[:, j] = (Fx_p - Fx) / h
            else:
                J[:, j] = 1.0  # Fallback
        except Exception:
            J[:, j] = 1.0
    
    if calls >= _CALL_LIMIT:
        return best_x
    
    # Broyden iteration
    s_prev = None
    y_prev = None
    max_iters = 480
    
    for it in range(max_iters):
        if calls >= _CALL_LIMIT or not _elapsed_ok(start):
            break
        
        # Newton step with current Jacobian estimate
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular or ill-conditioned, try pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, Fx, rcond=None)[0]
            except Exception:
                break
        except Exception:
            break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Trust region damping
        norm_dx = _norm2(dx)
        rho = max(1.0, 0.5 * _norm2(x))
        if norm_dx > rho:
            dx = dx * (rho / norm_dx)
        
        x_new = x + dx
        if not _finite2(x_new):
            # Halve the step
            dx = dx * 0.5
            x_new = x + dx
            if not _finite2(x_new):
                break
        
        # Evaluate F at new point
        try:
            Fx_new = np.asarray(F(x_new), dtype=float)
            calls += 1
        except Exception:
            break
        
        if not _finite2(Fx_new):
            # Try smaller step
            dx = dx * 0.5
            x_new = x + dx
            if not _finite2(x_new):
                break
            try:
                Fx_new = np.asarray(F(x_new), dtype=float)
                calls += 1
            except Exception:
                break
            if not _finite2(Fx_new):
                break
        
        norm_new = _norm2(Fx_new)
        
        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        
        if np.all(np.isfinite(s)) and np.all(np.isfinite(y)):
            # Standard Broyden update: J_new = J + (y - J*s) * s^T / (s^T * s)
            Js = J @ s
            diff = y - Js
            sTs = s @ s
            if sTs > 1e-12:
                J = J + np.outer(diff, s) / sTs
        
        # Track best
        if norm_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_new
        
        # Check convergence
        if best_norm <= 1e-8:
            break
        
        # Update state
        x = x_new
        Fx = Fx_new
        
        # Adaptive step size for FD if we need to reinitialize
        # (not typically needed with Broyden)
        
        # If we're stuck, try a small random perturbation
        if it > 50 and norm_new >= best_norm * 0.99:
            # Check if we're oscillating
            if _norm2(x - best_x) < 1e-6:
                # Try a small perturbation
                pert = np.random.randn(2) * 0.1
                x_pert = x + pert
                if _finite2(x_pert) and calls < _CALL_LIMIT - 1:
                    try:
                        Fx_pert = np.asarray(F(x_pert), dtype=float)
                        calls += 1
                        if _finite2(Fx_pert):
                            norm_pert = _norm2(Fx_pert)
                            if norm_pert < best_norm:
                                x = x_pert
                                Fx = Fx_pert
                                best_x = x.copy()
                                best_norm = norm_pert
                    except Exception:
                        pass
    
    return best_x
