"""Root finder for 3 fixed 2-D systems using Broyden's method."""
import time
import numpy as np


def find_root_system(F, x0):
    deadline = time.time() + 145.0  # safety margin before 150s kill
    tol = 1e-8
    x = np.array(x0, dtype=float)
    
    def time_ok():
        return time.time() < deadline
    
    def safe_eval_F(v):
        """Evaluate F with safety checks."""
        try:
            result = np.asarray(F(v), dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None
    
    # Initial evaluation
    Fx = safe_eval_F(x)
    if Fx is None:
        return x
    if np.linalg.norm(Fx) <= tol:
        return x
    
    # Compute initial Jacobian via finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = safe_eval_F(x + e)
        if Fx_plus is None:
            return x
        J[:, j] = (Fx_plus - Fx) / h
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    # Broyden's good method
    max_iters = 480  # leave room for line search calls
    for it in range(max_iters):
        if not time_ok():
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            break
        
        # Try full step first
        x_new = x + dx
        Fx_new = safe_eval_F(x_new)
        if Fx_new is None:
            break
        
        norm_new = np.linalg.norm(Fx_new)
        
        if norm_new <= tol:
            return x_new
        
        # Check if step improved
        if norm_new < best_norm:
            # Update Jacobian (Broyden's good method)
            s = x_new - x
            y = Fx_new - Fx
            # J = J + (y - J*s) * s^T / (s^T * s)
            Js = J @ s
            J += np.outer(y - Js, s) / (s @ s)
            
            x = x_new
            Fx = Fx_new
            best_x = x.copy()
            best_norm = norm_new
        else:
            # Try damped step (backtracking)
            alpha = 0.5
            found = False
            for _ in range(5):
                if not time_ok():
                    break
                x_trial = x + alpha * dx
                Fx_trial = safe_eval_F(x_trial)
                if Fx_trial is None:
                    break
                norm_trial = np.linalg.norm(Fx_trial)
                if norm_trial < best_norm:
                    # Update Jacobian
                    s = x_trial - x
                    y = Fx_trial - Fx
                    Js = J @ s
                    J += np.outer(y - Js, s) / (s @ s)
                    
                    x = x_trial
                    Fx = Fx_trial
                    best_x = x.copy()
                    best_norm = norm_trial
                    if norm_trial <= tol:
                        return x
                    found = True
                    break
                alpha *= 0.5
            if not found:
                # Stuck, break
                break
    
    return best_x
