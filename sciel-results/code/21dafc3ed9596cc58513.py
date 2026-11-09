"""Root finder for 3 fixed two-dimensional systems using derivative-free trust-region + Broyden polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 145.0  # Safety margin before 150s kill


def _time_ok():
    return (time.time() - _START_TIME) < _DEADLINE


def _is_finite(arr):
    try:
        return arr.shape == (2,) and np.all(np.isfinite(arr))
    except Exception:
        return False


def _check_solution(F, x):
    """Check if x is a valid solution with ||F(x)|| <= 1e-8."""
    if not _is_finite(x):
        return False
    Fx = F(x)
    if not _is_finite(Fx):
        return False
    return np.linalg.norm(Fx) <= 1e-8


def _newton_polish(F, x, max_iter=10):
    """Newton's method with FD Jacobian for final polish."""
    x = np.array(x, dtype=float)
    Fx = F(x)
    if not _is_finite(Fx):
        return x
    
    for _ in range(max_iter):
        if not _time_ok():
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            break
        
        # Finite difference Jacobian (central differences for better accuracy)
        h = 1e-8 * max(1.0, np.linalg.norm(x))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_p = F(x + e)
            Fx_m = F(x - e)
            if _is_finite(Fx_p) and _is_finite(Fx_m):
                J[:, j] = (Fx_p - Fx_m) / (2 * h)
            else:
                # Fallback to forward difference
                if _is_finite(Fx_p):
                    J[:, j] = (Fx_p - Fx) / h
                elif _is_finite(Fx_m):
                    J[:, j] = (Fx - Fx_m) / h
                else:
                    return x
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            return x
        
        x_new = x + dx
        Fx_new = F(x_new)
        if not _is_finite(Fx_new):
            break
        
        if np.linalg.norm(Fx_new) > norm_Fx:
            # Damp the step
            for damp in [0.5, 0.25, 0.1]:
                x_new = x + damp * dx
                Fx_new = F(x_new)
                if _is_finite(Fx_new) and np.linalg.norm(Fx_new) < norm_Fx:
                    break
            else:
                break
        
        x = x_new
        Fx = Fx_new
    
    return x


def _broyden(F, x0, max_iter=50):
    """Broyden's method for fast convergence on well-conditioned systems."""
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if not _is_finite(Fx):
        return x
    
    # Initial Jacobian via forward differences
    eps = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_e = F(x + e)
        if _is_finite(Fx_e):
            J[:, j] = (Fx_e - Fx) / eps
        else:
            J[:, j] = np.ones(2)
    
    for _ in range(max_iter):
        if not _time_ok():
            break
        
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-12:
            break
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            J[0, 0] += 1e-10
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
        
        # Simple backtracking line search
        alpha = 1.0
        x_new = x + alpha * dx
        Fx_new = F(x_new)
        
        while not _is_finite(Fx_new) or np.linalg.norm(Fx_new) > (1 - 1e-4 * alpha) * norm_Fx:
            alpha *= 0.5
            if alpha < 1e-12:
                if _is_finite(Fx_new):
                    break
                else:
                    return x
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            if not _time_ok():
                break
        
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = s @ s
        
        if s_norm_sq > 1e-14:
            Js = J @ s
            diff = Js - y
            J = J - np.outer(diff, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
    
    return x


def _dogleg_solve(F, x0, max_nfev=200):
    """Derivative-free trust-region via dogleg on ||F||^2."""
    x0 = np.array(x0, dtype=float)
    
    def objective(x):
        Fx = F(x)
        if not _is_finite(Fx):
            return 1e30
        return float(np.dot(Fx, Fx))
    
    try:
        result = minimize(objective, x0, method='dogleg', options={'maxiter': 50, 'max_nfev': max_nfev})
        if result.success or _check_solution(F, result.x):
            if _is_finite(result.x):
                Fx = F(result.x)
                if _is_finite(Fx) and np.linalg.norm(Fx) <= 1e-8:
                    return result.x
    except Exception:
        pass
    
    return None


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Quick check if x0 is already a solution
    2. Try Broyden's method (fast, few calls for well-conditioned systems)
    3. If Broyden converges to high precision, polish with Newton
    4. If Broyden fails, try dogleg (derivative-free, robust)
    5. Multi-start with small perturbations if needed
    """
    x0 = np.array(x0, dtype=float)
    
    # Quick check
    if _check_solution(F, x0):
        return x0
    
    # Step 1: Try Broyden from x0
    x_broyden = _broyden(F, x0, max_iter=30)
    if _check_solution(F, x_broyden):
        # Polish with Newton for extra precision
        x_polished = _newton_polish(F, x_broyden)
        if _check_solution(F, x_polished):
            return x_polished
        return x_broyden
    
    # Step 2: Try dogleg (derivative-free, more robust)
    x_dogleg = _dogleg_solve(F, x0, max_nfev=150)
    if x_dogleg is not None:
        x_polished = _newton_polish(F, x_dogleg)
        if _check_solution(F, x_polished):
            return x_polished
        return x_dogleg
    
    # Step 3: Multi-start with Broyden
    if _time_ok():
        for i in range(8):
            if not _time_ok():
                break
            angle = 2 * np.pi * i / 8
            radius = 0.05 * (i + 1)
            perturbation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_start = x0 + perturbation
            
            x_trial = _broyden(F, x_start, max_iter=20)
            if _check_solution(F, x_trial):
                x_polished = _newton_polish(F, x_trial)
                if _check_solution(F, x_polished):
                    return x_polished
                return x_trial
    
    # Step 4: Multi-start with dogleg
    if _time_ok():
        for i in range(6):
            if not _time_ok():
                break
            angle = 2 * np.pi * i / 6
            radius = 0.1 * (i + 1)
            perturbation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_start = x0 + perturbation
            
            x_trial = _dogleg_solve(F, x_start, max_nfev=80)
            if x_trial is not None:
                x_polished = _newton_polish(F, x_trial)
                if _check_solution(F, x_polished):
                    return x_polished
                return x_trial
    
    # Step 5: Extended Broyden with more iterations
    if _time_ok():
        x_ext = _broyden(F, x0, max_iter=80)
        if _check_solution(F, x_ext):
            x_polished = _newton_polish(F, x_ext)
            if _check_solution(F, x_polished):
                return x_polished
            return x_ext
    
    # Step 6: Larger radius multi-start
    if _time_ok():
        for i in range(10):
            if not _time_ok():
                break
            angle = 2 * np.pi * i / 10
            radius = 0.5 * (i + 1)
            perturbation = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            x_start = x0 + perturbation
            
            x_trial = _broyden(F, x_start, max_iter=30)
            if _check_solution(F, x_trial):
                x_polished = _newton_polish(F, x_trial)
                if _check_solution(F, x_polished):
                    return x_polished
                return x_trial
    
    # Last resort: return best effort
    Fx0 = F(x0)
    if _is_finite(Fx0):
        return x0
    
    return x0
# EVOLVE-BLOCK-END
