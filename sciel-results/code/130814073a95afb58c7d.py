"""Root finder for 3 fixed two-dimensional systems using Broyden's method with line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Broyden's method with backtracking line search.
    Returns x such that ||F(x)|| <= 1e-8.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = np.array(x0, dtype=float).copy()
    
    # Initial function evaluation
    try:
        Fx = np.asarray(F(x), dtype=float)
        if Fx.shape != (2,) or not np.all(np.isfinite(Fx)):
            raise ValueError("Invalid F output")
    except Exception:
        return x
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian with finite differences (2 extra calls)
    J = np.zeros((2, 2))
    h = 1e-6
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        try:
            Fx_pert = np.asarray(F(x + e), dtype=float)
            if Fx_pert.shape != (2,) or not np.all(np.isfinite(Fx_pert)):
                raise ValueError("Invalid F output")
            J[:, j] = (Fx_pert - Fx) / h
        except Exception:
            # If FD fails, use identity
            J = np.eye(2)
            break
    
    best_x = x.copy()
    best_Fx = Fx.copy()
    best_norm = np.linalg.norm(Fx)
    
    max_iter = 100
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            return best_x
        
        # Compute search direction: solve J * s = -Fx
        try:
            s = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Fallback: steepest descent direction
            s = -Fx
            if np.linalg.norm(s) < 1e-14:
                break
        
        # Backtracking line search
        alpha = 1.0
        Fx_old = Fx.copy()
        x_old = x.copy()
        c1 = 1e-4
        rho = 0.5
        
        found_step = False
        for ls_iter in range(30):
            if time.time() > deadline:
                break
            
            x_trial = x_old + alpha * s
            try:
                Fx_trial = np.asarray(F(x_trial), dtype=float)
                if Fx_trial.shape != (2,) or not np.all(np.isfinite(Fx_trial)):
                    raise ValueError("Invalid F output")
            except Exception:
                alpha *= rho
                continue
            
            norm_trial = np.linalg.norm(Fx_trial)
            norm_old = np.linalg.norm(Fx_old)
            
            # Armijo condition: sufficient decrease
            if norm_trial <= norm_old - c1 * alpha * np.dot(Fx_old, s):
                found_step = True
                break
            
            alpha *= rho
        
        if not found_step:
            # Try a small damped step
            alpha = 1e-4
            x_trial = x_old + alpha * s
            try:
                Fx_trial = np.asarray(F(x_trial), dtype=float)
                if Fx_trial.shape != (2,) or not np.all(np.isfinite(Fx_trial)):
                    break
            except Exception:
                break
            
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_Fx = Fx_trial.copy()
                best_norm = norm_trial
            
            # Update Broyden
            if np.linalg.norm(x_trial - x_old) > 1e-14:
                s_vec = x_trial - x_old
                y_vec = Fx_trial - Fx_old
                denom = np.dot(y_vec, s_vec)
                if abs(denom) > 1e-14:
                    J = J + np.outer(y_vec - J @ s_vec, s_vec) / denom
            
            x = x_trial
            Fx = Fx_trial
            continue
        
        # Accept the step
        x = x_trial
        Fx = Fx_trial
        
        # Update best
        norm_current = np.linalg.norm(Fx)
        if norm_current < best_norm:
            best_x = x.copy()
            best_Fx = Fx.copy()
            best_norm = norm_current
        
        # Broyden rank-1 update
        s_vec = x - x_old
        y_vec = Fx - Fx_old
        denom = np.dot(y_vec, s_vec)
        if abs(denom) > 1e-14:
            J = J + np.outer(y_vec - J @ s_vec, s_vec) / denom
    
    # Return the best point found
    return best_x
# EVOLVE-BLOCK-END
