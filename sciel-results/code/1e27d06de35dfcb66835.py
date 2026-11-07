# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root as scipy_root


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses a multi-strategy approach:
    1. Broyden's method with adaptive line search
    2. Fallback to SciPy's hybrid (Powell) method if Broyden fails
    3. Time guard to ensure we return within deadline
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def norm(v):
        return np.linalg.norm(v)
    
    def check_convergence(x, Fx):
        if not np.all(np.isfinite(Fx)):
            return False
        return norm(Fx) <= 1e-8
    
    def is_valid_point(x):
        return np.all(np.isfinite(x)) and np.all(np.abs(x) < 1e10)
    
    # Phase 1: Broyden's method
    try:
        x_broyden = _broyden(F, x, deadline, time_up)
        if x_broyden is not None:
            Fx = F(x_broyden)
            if check_convergence(x_broyden, Fx):
                return x_broyden
    except Exception:
        pass
    
    # Phase 2: Try SciPy's hybrid method from original x0
    if not time_up():
        try:
            res = scipy_root(F, x, method='hybrid', options={'maxiter': 200, 'xtol': 1e-12, 'ftol': 1e-12})
            if res.success and is_valid_point(res.x):
                Fx = F(res.x)
                if check_convergence(res.x, Fx):
                    return res.x
            # Even if not fully converged, if close enough
            if is_valid_point(res.x):
                Fx = F(res.x)
                if norm(Fx) <= 1e-8:
                    return res.x
        except Exception:
            pass
    
    # Phase 3: Try SciPy from perturbed points
    if not time_up():
        perturbations = [
            np.array([0.01, 0.0]),
            np.array([0.0, 0.01]),
            np.array([0.01, 0.01]),
            np.array([-0.01, 0.0]),
            np.array([0.0, -0.01]),
            np.array([-0.01, -0.01]),
            np.array([0.1, 0.0]),
            np.array([0.0, 0.1]),
        ]
        for p in perturbations:
            if time_up():
                break
            x_try = x + p
            if not is_valid_point(x_try):
                continue
            try:
                res = scipy_root(F, x_try, method='hybrid', options={'maxiter': 100, 'xtol': 1e-12, 'ftol': 1e-12})
                if is_valid_point(res.x):
                    Fx = F(res.x)
                    if check_convergence(res.x, Fx):
                        return res.x
            except Exception:
                pass
    
    # Phase 4: Return best candidate found
    # Evaluate original x0 and return it (shouldn't happen if all failed)
    return x


def _broyden(F, x0, deadline, time_up):
    """Broyden's method with adaptive line search."""
    x = np.array(x0, dtype=float).copy()
    
    # Initial function evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        x = x + np.array([1e-4, 1e-4])
        Fx = F(x)
        if not np.all(np.isfinite(Fx)):
            return None
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Initial Jacobian: finite difference
    J = _finite_diff_jacobian(F, x, Fx)
    if J is None:
        return None
    
    # Broyden iterations
    max_iter = 100
    for it in range(max_iter):
        if time_up():
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular matrix, try pseudo-inverse
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except Exception:
                break
        
        if not np.all(np.isfinite(dx)):
            break
        
        # Line search
        alpha = 1.0
        x_new = x
        Fx_new = Fx
        for ls_iter in range(20):
            if time_up():
                break
            x_new = x + alpha * dx
            if not np.all(np.isfinite(x_new)):
                alpha *= 0.5
                continue
            
            Fx_new = F(x_new)
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            if np.linalg.norm(Fx_new) <= 1e-8:
                return x_new
            
            # Check if this step reduces the norm
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                break
            
            alpha *= 0.5
        
        # Update Jacobian using Broyden's formula
        delta_x = x_new - x
        delta_F = Fx_new - Fx
        
        if np.linalg.norm(delta_x) < 1e-15:
            break
        
        J = J + np.outer(delta_F - J @ delta_x, delta_x) / (delta_x @ delta_x)
        
        x = x_new
        Fx = Fx_new
    
    return x


def _finite_diff_jacobian(F, x, Fx, eps=1e-7):
    """Compute Jacobian via forward differences."""
    n = len(x)
    J = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = eps
        Fx_pert = F(x + e)
        if not np.all(np.isfinite(Fx_pert)):
            return None
        J[:, j] = (Fx_pert - Fx) / eps
    return J
# EVOLVE-BLOCK-END
