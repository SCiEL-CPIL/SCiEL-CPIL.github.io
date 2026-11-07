# EVOLVE-BLOCK-START
import numpy as np


def _newton_solve(F, x, max_iter=40, tol=1e-13, eps=1e-7):
    """Newton's method with finite-difference Jacobian for 2D system."""
    x = np.array(x, dtype=float)
    for _ in range(max_iter):
        Fx = np.asarray(F(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= tol:
            return x
        # Compute Jacobian via forward differences
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = eps
            Fx_e = np.asarray(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fx_e)):
                # Try smaller step
                eps2 = eps * 0.1
                e[j] = eps2
                Fx_e = np.asarray(F(x + e), dtype=float)
                if not np.all(np.isfinite(Fx_e)):
                    return x
                J[:, j] = (Fx_e - Fx) / eps2
            else:
                J[:, j] = (Fx_e - Fx) / eps
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Try pseudo-inverse
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                return x
        if not np.all(np.isfinite(dx)):
            return x
        # Damping: reduce step if it increases norm
        step = 1.0
        for _ in range(20):
            x_new = x + step * dx
            Fx_new = np.asarray(F(x_new), dtype=float)
            if np.all(np.isfinite(Fx_new)):
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx * 1.01:
                    x = x_new
                    break
            step *= 0.5
        else:
            return x
    return x


def find_root_system(F, x0):
    """
    Homotopy/continuation method to find a root of F(x) = 0.
    
    Solves F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1,
    tracking the root path. At t=0, the unique solution is x=x0.
    We continue t to 1, where F_1(x) = F(x).
    """
    x = np.array(x0, dtype=float)
    
    # Homotopy function
    def F_t(x_val, t):
        Fx = np.asarray(F(x_val), dtype=float)
        return t * Fx + (1.0 - t) * (x_val - x)
    
    # We'll use a continuation scheme: increase t in steps, solving at each step
    # Use adaptive step size for t
    t = 0.0
    x_current = np.array(x0, dtype=float)
    
    # First, verify x0 is a root of F_0 (it should be trivially)
    # F_0(x) = x - x0, so root is x = x0
    
    # Continue t from 0 to 1
    max_t_steps = 50
    t_step = 0.05  # initial step size
    
    for t_iter in range(max_t_steps):
        t_new = min(t + t_step, 1.0)
        
        # Solve F_{t_new}(x) = 0 starting from x_current
        def F_t_new(x_val):
            Fx = np.asarray(F(x_val), dtype=float)
            return t_new * Fx + (1.0 - t_new) * (x_val - x)
        
        x_candidate = _newton_solve(F_t_new, x_current, max_iter=30, tol=1e-12, eps=1e-7)
        
        # Check if we got a good solution
        F_candidate = F_t_new(x_candidate)
        if np.all(np.isfinite(F_candidate)) and np.linalg.norm(F_candidate) < 1e-10:
            x_current = x_candidate
            t = t_new
            # If we've reached t=1, we're done
            if t >= 1.0:
                break
            # Increase step size if we converged quickly
            t_step = min(t_step * 1.5, 0.2)
        else:
            # Reduce step size and retry
            t_step *= 0.5
            if t_step < 1e-4:
                # Try a final push with smaller steps
                remaining = 1.0 - t
                n_sub = 10
                for i in range(n_sub):
                    t_sub = t + remaining * (i + 1) / n_sub
                    if t_sub >= 1.0:
                        t_sub = 1.0
                    
                    def F_t_sub(x_val):
                        Fx = np.asarray(F(x_val), dtype=float)
                        return t_sub * Fx + (1.0 - t_sub) * (x_val - x)
                    
                    x_candidate = _newton_solve(F_t_sub, x_current, max_iter=20, tol=1e-11, eps=1e-7)
                    
                    F_candidate = F_t_sub(x_candidate)
                    if np.all(np.isfinite(F_candidate)) and np.linalg.norm(F_candidate) < 1e-10:
                        x_current = x_candidate
                        t = t_sub
                    else:
                        break
                break
    
    # Final check: is x_current a root of F?
    F_final = np.asarray(F(x_current), dtype=float)
    if np.all(np.isfinite(F_final)) and np.linalg.norm(F_final) <= 1e-8:
        return x_current
    
    # If homotopy didn't converge well, try direct Newton from x_current
    x_direct = _newton_solve(F, x_current, max_iter=40, tol=1e-13, eps=1e-7)
    F_direct = np.asarray(F(x_direct), dtype=float)
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) <= 1e-8:
        return x_direct
    
    # Try from x0 directly as fallback
    x_from_x0 = _newton_solve(F, x0, max_iter=40, tol=1e-13, eps=1e-7)
    F_from_x0 = np.asarray(F(x_from_x0), dtype=float)
    if np.all(np.isfinite(F_from_x0)) and np.linalg.norm(F_from_x0) <= 1e-8:
        return x_from_x0
    
    # Return the best candidate we have
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) < np.linalg.norm(F_final) if np.all(np.isfinite(F_final)) else True:
        return x_direct
    return x_current
# EVOLVE-BLOCK-END
