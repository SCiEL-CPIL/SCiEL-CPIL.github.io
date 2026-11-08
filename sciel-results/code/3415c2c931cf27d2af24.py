"""Root finder for 3 fixed two-dimensional systems using Broyden's method with safeguards."""
# EVOLVE-BLOCK-START
import numpy as np
import time


_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _broyden(F, x0, max_iter=50):
    """Broyden's method (good inverse) for 2D systems."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if not _is_finite(Fx):
        return x, 1
    
    # Initial Jacobian estimate using finite differences
    J = np.zeros((2, 2))
    eps = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_e = F(x + e)
        if not _is_finite(Fx_e):
            Fx_e = Fx + eps * np.ones(2)  # fallback
        J[:, j] = (Fx_e - Fx) / eps
    
    calls = 1 + 2  # 1 for Fx, 2 for Jacobian
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        # Check convergence
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x, calls
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, perturb it
            J[0, 0] += 1e-10
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Line search with backtracking
        alpha = 1.0
        x_new = x + alpha * dx
        Fx_new = F(x_new)
        calls += 1
        
        while not _is_finite(Fx_new) or np.linalg.norm(Fx_new) > (1 - 1e-4 * alpha) * norm_Fx:
            alpha *= 0.5
            if alpha < 1e-10:
                # Accept the step anyway if it's finite
                if _is_finite(Fx_new):
                    break
                else:
                    # Revert to previous x
                    return x, calls
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            calls += 1
            if not _time_ok():
                break
        
        # Broyden update
        s = x_new - x
        y = Fx_new - Fx
        
        if np.linalg.norm(s) > 1e-14:
            # Good inverse update: J_new = J - (J*s - y) * s^T / (s^T * s)
            Js = J @ s
            diff = Js - y
            J = J - np.outer(diff, s) / (s @ s)
        
        x = x_new
        Fx = Fx_new
    
    return x, calls


def _newton_fd(F, x0, max_iter=30):
    """Newton's method with finite-difference Jacobian as fallback."""
    x = np.array(x0, dtype=float)
    calls = 0
    
    for it in range(max_iter):
        if not _time_ok():
            break
        
        Fx = F(x)
        calls += 1
        if not _is_finite(Fx):
            return x, calls
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            return x, calls
        
        J = np.zeros((2, 2))
        eps = 1e-6
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = F(x + e)
            calls += 1
            if not _isfinite_check(Fx_e):
                Fx_e = Fx + eps
            J[:, j] = (Fx_e - Fx) / eps
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        x = x + dx
    
    return x, calls


def _isfinite_check(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method as primary, Newton FD as fallback.
    """
    x0 = np.array(x0, dtype=float)
    
    # Try Broyden's method first
    x_broyden, calls_broyden = _broyden(F, x0)
    
    # Verify the result
    if _is_finite(x_broyden):
        Fx_check = F(x_broyden)
        if _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_broyden
    
    # If Broyden didn't converge, try Newton FD
    x_newton, calls_newton = _newton_fd(F, x0)
    
    if _is_finite(x_newton):
        Fx_check = F(x_newton)
        if _is_finite(Fx_check) and np.linalg.norm(Fx_check) <= 1e-8:
            return x_newton
    
    # Return the better of the two
    if _is_finite(x_broyden) and _is_finite(x_newton):
        Fx_b = F(x_broyden)
        Fx_n = F(x_newton)
        if _is_finite(Fx_b) and _is_finite(Fx_n):
            if np.linalg.norm(Fx_b) <= np.linalg.norm(Fx_n):
                return x_broyden
            else:
                return x_newton
        elif _is_finite(Fx_b):
            return x_broyden
        else:
            return x_newton
    elif _is_finite(x_broyden):
        return x_broyden
    else:
        return x_newton
# EVOLVE-BLOCK-END
