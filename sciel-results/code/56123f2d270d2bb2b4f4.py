"""Root finder for 3 fixed two-dimensional systems using Broyden's method."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import root

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Use Broyden's method (1 F call/iter after setup) with multiple restarts.
    2. Broyden updates Jacobian approximation, so each iteration costs 1 F(x) call.
    3. For 2D, initial Jacobian needs 3 calls (F(x0), F(x0+e1), F(x0+e2)).
    4. Fallback to scipy's hybrid method if Broyden fails.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    x0 = np.array(x0, dtype=float)
    
    def is_good(x):
        try:
            if x is None:
                return False
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return False
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return False
            return np.linalg.norm(fx) <= 1e-8
        except:
            return False
            
    def time_ok():
        return time.time() < deadline

    # Broyden's method implementation
    def broyden_method(x_init, max_iter=50):
        x = np.array(x_init, dtype=float)
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
                return x
            n = np.linalg.norm(Fx)
            if n <= 1e-10:
                return x
            
            # Initialize Jacobian
            J = np.zeros((2, 2))
            h = 1e-8
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            
            Fx1 = F(x + e1)
            Fx1 = np.asarray(Fx1, dtype=float)
            J[:, 0] = (Fx1 - Fx) / h
            
            Fx2 = F(x + e2)
            Fx2 = np.asarray(Fx2, dtype=float)
            J[:, 1] = (Fx2 - Fx) / h
            
            for i in range(max_iter):
                if not time_ok():
                    return x
                
                try:
                    # Solve J * dx = -Fx
                    try:
                        dx = np.linalg.solve(J, -Fx)
                    except np.linalg.LinAlgError:
                        # Regularize if singular
                        try:
                            dx = np.linalg.solve(J + 1e-10 * np.eye(2), -Fx)
                        except:
                            break
                    
                    # Limit step size
                    step_norm = np.linalg.norm(dx)
                    if step_norm > 10.0:
                        dx = dx * (10.0 / step_norm)
                    
                    x_new = x + dx
                    if not np.all(np.isfinite(x_new)):
                        break
                    
                    Fx_new = F(x_new)
                    Fx_new = np.asarray(Fx_new, dtype=float)
                    n_new = np.linalg.norm(Fx_new)
                    
                    if not np.isfinite(n_new):
                        break
                    
                    if n_new <= 1e-10:
                        return x_new
                    
                    # Broyden update
                    s = x_new - x
                    y = Fx_new - Fx
                    
                    # Check for rank-1 update validity
                    if np.linalg.norm(s) > 1e-12:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    
                    x = x_new
                    Fx = Fx_new
                    n = n_new
                    
                except:
                    break
            
            return x
        except:
            return x

    # Initial points
    initial_points = [x0.copy()]
    scale = max(1.0, np.linalg.norm(x0))
    # Fewer, smarter perturbations
    perturbations = [
        0.1 * scale, 0.5 * scale, 2.0 * scale
    ]
    for s in perturbations:
        initial_points.append(x0 + np.array([s, 0.0]))
        initial_points.append(x0 + np.array([0.0, s]))
        initial_points.append(x0 + np.array([s, s]) / np.sqrt(2))
        initial_points.append(x0 - np.array([s, 0.0]))
        initial_points.append(x0 - np.array([0.0, s]))

    best_x = x0.copy()
    best_norm = np.inf

    def update_best(x):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass

    # Try Broyden from each initial point
    for x_init in initial_points:
        if not time_ok():
            break
        
        x_result = broyden_method(x_init, max_iter=30)
        if x_result is not None:
            update_best(x_result)
            if best_norm <= 1e-8:
                return best_x

    # Fallback to scipy hybrid if Broyden didn't succeed
    if not time_ok():
        return best_x if np.all(np.isfinite(best_x)) else x0

    # Try scipy hybrid from the best point found
    try:
        sol = root(lambda x: np.asarray(F(x), dtype=float), best_x, method='hybrid', tol=1e-12)
        if sol.success or np.linalg.norm(sol.fun) < 1e-8:
            return sol.x
    except:
        pass

    # Final fallback: return best found
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
