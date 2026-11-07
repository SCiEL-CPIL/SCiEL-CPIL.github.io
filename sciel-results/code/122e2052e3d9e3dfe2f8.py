"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with x0 and use Broyden's method (good initial Jacobian estimate).
    2. If stuck, try a few random restarts from scaled versions of x0.
    3. Use damping if step is too large.
    
    Budget:
    - Time limit: ~115s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            result = F(np.array(x, dtype=float))
            result = np.array(result, dtype=float)
            if result.shape != (2,):
                result = result.reshape(2)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except:
            return None
    
    def broyden(x0, max_iter=50, max_calls=450):
        """
        Broyden's method with safeguards.
        Returns (x, Fx, num_calls, success).
        """
        x = np.array(x0, dtype=float)
        calls = 0
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, calls, False
        calls += 1
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, calls, True
        
        # Initial Jacobian: use finite differences (2 extra calls for 2D)
        # Or start with identity - but finite diff is better for convergence
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            if calls >= max_calls or not time_ok():
                break
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                J[:, j] = 1.0 if j == 0 else 0.0  # fallback
            else:
                J[:, j] = (Fx_e - Fx) / h
            calls += 1
        
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
        
        # Broyden's good method iteration
        for it in range(max_iter):
            if calls >= max_calls or not time_ok():
                break
            
            # Check convergence
            if np.linalg.norm(Fx) <= 1e-8:
                return x, Fx, calls, True
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Damping: if step is too large, shrink it
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Try smaller step
                dx = dx * 0.5
                x_new = x + dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break
            calls += 1
            
            # Check if improvement
            if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                # Update Jacobian (Broyden's good method)
                s = x_new - x
                y = Fx_new - Fx
                try:
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                except:
                    pass
                x = x_new
                Fx = Fx_new
            else:
                # No improvement, try smaller step
                dx = dx * 0.5
                x_new = x + dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    break
                calls += 1
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    s = x_new - x
                    y = Fx_new - Fx
                    try:
                        J = J + np.outer(y - J @ s, s) / (s @ s)
                    except:
                        pass
                    x = x_new
                    Fx = Fx_new
                else:
                    # Stuck, break
                    break
        
        return x, Fx, calls, (Fx is not None and np.linalg.norm(Fx) <= 1e-8)
    
    # First try from x0
    x, Fx, calls, success = broyden(x0, max_iter=40, max_calls=450)
    if success:
        return x
    
    # If failed, try a few restarts with different initial points
    # Use scaled versions of x0 and some heuristic offsets
    best_x = x
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    restarts = []
    x0_arr = np.array(x0, dtype=float)
    # Try x0 scaled by various factors
    for scale in [0.5, 2.0, 0.1, 5.0]:
        restarts.append(x0_arr * scale)
    # Try small perturbations
    for axis in range(2):
        for delta in [-1.0, 1.0]:
            pt = x0_arr.copy()
            pt[axis] += delta
            restarts.append(pt)
    
    for restart_pt in restarts:
        if not time_ok() or calls >= 490:
            break
        x_r, Fx_r, calls_r, success_r = broyden(restart_pt, max_iter=30, max_calls=490 - calls)
        calls += calls_r
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_x = x_r
                best_norm = norm_r
            if success_r:
                return x_r
    
    return best_x
