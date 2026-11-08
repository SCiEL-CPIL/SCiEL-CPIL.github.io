"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        arr = np.asarray(arr, dtype=float)
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _fd_jacobian(F, x):
    """Compute finite-difference Jacobian using central differences."""
    J = np.zeros((2, 2))
    h = 1e-6
    Fx = F(x)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def _fd_jacobian_tight(F, x):
    """Compute finite-difference Jacobian with tighter step for polishing."""
    J = np.zeros((2, 2))
    h = 1e-8
    Fx = F(x)
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - F(x - e)) / (2 * h)
    return J


def _newton_polish(F, x, Fx, norm_Fx, tol=1e-12, max_iter=10):
    """High-precision Newton polish pass with tighter FD Jacobian."""
    best_x = x.copy()
    best_norm = norm_Fx
    for _ in range(max_iter):
        if not _time_ok():
            break
        if norm_Fx <= tol:
            break
        try:
            J = _fd_jacobian_tight(F, x)
        except Exception:
            break
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(dx)):
            break
        # Full Newton step (no line search) for local quadratic convergence
        x_trial = x + dx
        try:
            Fx_trial = np.asarray(F(x_trial), dtype=float)
        except Exception:
            break
        if not _is_finite(Fx_trial):
            break
        norm_trial = np.linalg.norm(Fx_trial)
        if norm_trial < norm_Fx:
            x = x_trial
            Fx = Fx_trial
            norm_Fx = norm_trial
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
    return best_x, best_norm


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8 using Broyden's method
    with backtracking line search and restarts.
    """
    x = np.array(x0, dtype=float)
    
    # Initial function evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
    except Exception:
        return x
    
    if not _is_finite(Fx):
        return x
    
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-12:
        return x
    
    best_x = x.copy()
    best_norm = norm_Fx
    
    # Initialize Jacobian with finite differences
    try:
        J = _fd_jacobian(F, x)
    except Exception:
        J = np.eye(2)
    
    # Broyden's method iterations
    max_iter = 80
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Check convergence
        if norm_Fx <= 1e-12:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Restart with fresh Jacobian
            try:
                J = _fd_jacobian(F, x)
            except Exception:
                J = np.eye(2)
            continue
        
        # Backtracking line search
        alpha = 1.0
        x_trial = x + alpha * dx
        Fx_trial = None
        for _ in range(20):
            if not _time_ok():
                break
            try:
                x_trial = x + alpha * dx
                Fx_trial = np.asarray(F(x_trial), dtype=float)
            except Exception:
                alpha *= 0.5
                continue
            
            if not _is_finite(Fx_trial):
                alpha *= 0.5
                continue
            
            norm_trial = np.linalg.norm(Fx_trial)
            
            if norm_trial < norm_Fx * (1 - 1e-4 * alpha):
                # Accept step
                s = x_trial - x
                y = Fx_trial - Fx
                x = x_trial
                Fx = Fx_trial
                norm_Fx = norm_trial
                
                # Update Jacobian (Broyden update)
                if np.linalg.norm(s) > 1e-14:
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                
                if norm_Fx < best_norm:
                    best_norm = norm_Fx
                    best_x = x.copy()
                
                if norm_Fx <= 1e-12:
                    break
                break
            else:
                alpha *= 0.5
        
        if norm_Fx <= 1e-12:
            break
    
    # If Broyden didn't converge well enough, try Newton with line search
    if best_norm > 1e-8 and _time_ok():
        x = best_x.copy()
        try:
            Fx = np.asarray(F(x), dtype=float)
        except Exception:
            Fx = np.array([0.0, 0.0])
        
        norm_Fx = np.linalg.norm(Fx) if _is_finite(Fx) else np.inf
        
        for it in range(30):
            if not _time_ok():
                break
            
            if norm_Fx <= 1e-12:
                break
            
            try:
                J = _fd_jacobian(F, x)
            except Exception:
                break
            
            try:
                dx = -np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                break
            
            if not np.all(np.isfinite(dx)):
                break
            
            # Backtracking line search
            alpha = 1.0
            accepted = False
            for _ in range(20):
                if not _time_ok():
                    break
                x_trial = x + alpha * dx
                try:
                    Fx_trial = np.asarray(F(x_trial), dtype=float)
                except Exception:
                    alpha *= 0.5
                    continue
                
                if not _is_finite(Fx_trial):
                    alpha *= 0.5
                    continue
                
                norm_trial = np.linalg.norm(Fx_trial)
                
                if norm_trial < norm_Fx * (1 - 1e-4 * alpha):
                    x = x_trial
                    Fx = Fx_trial
                    norm_Fx = norm_trial
                    if norm_Fx < best_norm:
                        best_norm = norm_Fx
                        best_x = x.copy()
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
    
    # Final high-precision polish pass
    if best_norm > 1e-12 and best_norm < 1e-6 and _time_ok():
        x_pol, norm_pol = _newton_polish(F, best_x, Fx, norm_Fx, tol=1e-12, max_iter=10)
        if norm_pol < best_norm:
            best_x = x_pol
            best_norm = norm_pol
    
    return best_x
# EVOLVE-BLOCK-END
