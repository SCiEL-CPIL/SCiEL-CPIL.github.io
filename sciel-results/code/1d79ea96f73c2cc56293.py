"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using a robust hybrid strategy:
    1. First try a damped Newton method with adaptive step size and finite-difference Jacobian.
    2. If Newton fails or stagnates, fall back to a global optimization approach
       (minimizing ||F(x)||^2) using Nelder-Mead or Powell from scipy.
    3. Finally, polish with a few Newton steps if close enough.
    
    The goal is to find x such that ||F(x)|| <= 1e-8 in Euclidean norm.
    """
    x = np.array(x0, dtype=float)
    
    # Check initial point
    try:
        Fx = F(x)
        if np.linalg.norm(Fx) <= 1e-8:
            return x
    except Exception:
        pass
    
    # Phase 1: Damped Newton with adaptive step size
    max_newton_steps = 30
    best_x = x.copy()
    best_norm = np.inf
    
    for it in range(max_newton_steps):
        try:
            Fx = F(x)
            norm_F = np.linalg.norm(Fx)
            if norm_F < best_norm:
                best_norm = norm_F
                best_x = x.copy()
            if norm_F <= 1e-10:
                return x
            
            # Compute finite-difference Jacobian
            eps = 1e-7
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                J[:, j] = (F(x + e) - Fx) / eps
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            # Adaptive step size: start with full step, reduce if norm doesn't decrease
            step = 1.0
            success = False
            for _ in range(15):
                x_new = x + step * dx
                try:
                    Fx_new = F(x_new)
                    norm_new = np.linalg.norm(Fx_new)
                except Exception:
                    step *= 0.5
                    continue
                
                if norm_new < norm_F:
                    x = x_new
                    success = True
                    break
                else:
                    step *= 0.5
            
            if not success:
                break
        except Exception:
            break
    
    # Check if we got close enough
    if best_norm <= 1e-8:
        return best_x
    
    # Phase 2: Global optimization using scipy.optimize.minimize
    # Minimize f(x) = ||F(x)||^2
    try:
        from scipy.optimize import minimize
        
        def objective(x):
            try:
                Fx = F(x)
                return np.dot(Fx, Fx)
            except Exception:
                return 1e10
        
        # Try multiple starting points: original x0, best_x, and some perturbations
        start_points = [best_x, x]
        # Add some random perturbations around best_x
        for _ in range(5):
            perturbation = 0.1 * np.random.randn(2)
            start_points.append(best_x + perturbation)
        
        best_result = None
        best_obj = np.inf
        
        for sp in start_points:
            try:
                result = minimize(objective, sp, method='Nelder-Mead',
                                 options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-20})
                if result.fun < best_obj:
                    best_obj = result.fun
                    best_result = result.x
            except Exception:
                continue
        
        if best_result is not None and best_obj <= 1e-16:
            # Polish with Newton
            x = best_result.copy()
            for _ in range(10):
                try:
                    Fx = F(x)
                    if np.linalg.norm(Fx) <= 1e-10:
                        return x
                    eps = 1e-7
                    J = np.zeros((2, 2))
                    for j in range(2):
                        e = np.zeros(2)
                        e[j] = eps
                        J[:, j] = (F(x + e) - Fx) / eps
                    dx = np.linalg.solve(J, -Fx)
                    x = x + dx
                except Exception:
                    break
            try:
                if np.linalg.norm(F(x)) <= 1e-8:
                    return x
            except Exception:
                pass
        
        if best_result is not None:
            return best_result
    except ImportError:
        pass
    except Exception:
        pass
    
    # Phase 3: Return the best point we found
    return best_x
# EVOLVE-BLOCK-END
