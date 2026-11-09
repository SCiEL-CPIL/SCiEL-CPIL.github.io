"""Root finder for 3 fixed two-dimensional systems using scipy's hybrid least-squares."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _solve_with_scipy(F, x0):
    """Use scipy's least_squares with 'lm' or 'trf' method."""
    x0 = np.array(x0, dtype=float)
    
    def fun(x):
        return F(x)
    
    # Try Levenberg-Marquardt first (fast for well-behaved problems)
    try:
        if _time_ok():
            result = least_squares(fun, x0, method='lm', max_nfev=200, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if _is_finite(result.x):
                Fx = F(result.x)
                if _is_finite(Fx) and np.linalg.norm(Fx) <= 1e-8:
                    return result.x
    except Exception:
        pass
    
    # Try Trust Region Reflective as fallback
    try:
        if _time_ok():
            result = least_squares(fun, x0, method='trf', max_nfev=500, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if _is_finite(result.x):
                Fx = F(result.x)
                if _is_finite(Fx) and np.linalg.norm(Fx) <= 1e-8:
                    return result.x
    except Exception:
        pass
    
    return None


def _broyden(F, x0, max_iter=100):
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


def _multi_start(F, x0, max_starts=10):
    """Try multiple starting points."""
    x0 = np.array(x0, dtype=float)
    best_x = x0
    best_norm = np.inf
    
    # Try original point
    result = _solve_with_scipy(F, x0)
    if result is not None:
        return result
    
    # Generate perturbed starting points
    for i in range(max_starts):
        if not _time_ok():
            break
        
        # Add small perturbations
        angle = 2 * np.pi * i / max_starts
        radius = 0.1 * (i + 1)
        perturbation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        x_start = x0 + perturbation
        
        result = _solve_with_scipy(F, x_start)
        if result is not None:
            return result
    
    # Fallback to Broyden with original point
    x_broyden, calls = _broyden(F, x0)
    if _is_finite(x_broyden):
        Fx = F(x_broyden)
        if _is_finite(Fx) and np.linalg.norm(Fx) <= 1e-8:
            return x_broyden
    
    return None


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses scipy's least_squares as primary, with multi-start and Broyden fallback.
    """
    x0 = np.array(x0, dtype=float)
    
    # First, try scipy's solver directly
    result = _solve_with_scipy(F, x0)
    if result is not None:
        return result
    
    # Try multi-start approach
    result = _multi_start(F, x0)
    if result is not None:
        return result
    
    # Last resort: Broyden with original point
    x_broyden, calls = _broyden(F, x0)
    if _is_finite(x_broyden):
        Fx = F(x_broyden)
        if _is_finite(Fx):
            return x_broyden
    
    # Return x0 if nothing works (shouldn't happen, but safety)
    return x0
# EVOLVE-BLOCK-END
