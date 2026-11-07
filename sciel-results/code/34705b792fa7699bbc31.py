# EVOLVE-BLOCK-START
import numpy as np


def _broyden_2d(F, x, max_iter=40, tol=1e-13, eps=1e-6):
    """Broyden's method for 2D systems with finite-difference Jacobian initialization.
    Uses rank-1 updates to approximate Jacobian, saving F calls."""
    x = np.array(x, dtype=float)
    Fx = np.asarray(F(x), dtype=float)
    if not np.all(np.isfinite(Fx)):
        return x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= tol:
        return x

    # Initialize Jacobian with finite differences
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        Fx_e = np.asarray(F(x + e), dtype=float)
        if not np.all(np.isfinite(Fx_e)):
            # Try smaller step
            eps2 = eps * 0.01
            e[j] = eps2
            Fx_e = np.asarray(F(x + e), dtype=float)
            if not np.all(np.isfinite(Fx_e)):
                return x
            J[:, j] = (Fx_e - Fx) / eps2
        else:
            J[:, j] = (Fx_e - Fx) / eps

    for _ in range(max_iter):
        # Solve J * dx = -Fx
        try:
            det = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
            if abs(det) < 1e-14:
                dx = np.linalg.pinv(J) @ (-Fx)
            else:
                dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except Exception:
                return x
        if not np.all(np.isfinite(dx)):
            return x

        # Line search with Armijo condition
        step = 1.0
        accepted = False
        for _ in range(20):
            x_new = x + step * dx
            Fx_new = np.asarray(F(x_new), dtype=float)
            if np.all(np.isfinite(Fx_new)):
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new <= norm_Fx - 1e-4 * step * np.linalg.norm(dx) * norm_Fx:
                    # Update Jacobian using Broyden's rank-1 update
                    s = x_new - x
                    y = Fx_new - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_new
                    Fx = Fx_new
                    norm_Fx = norm_Fx_new
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return x
        if norm_Fx <= tol:
            return x
    return x


def find_root_system(F, x0):
    """
    Homotopy/continuation method to find a root of F(x) = 0.
    
    Solves F_t(x) = t*F(x) + (1-t)*(x - x0) = 0 for t from 0 to 1,
    tracking the root path. At t=0, the unique solution is x=x0.
    We continue t to 1, where F_1(x) = F(x).
    
    Uses Broyden's method for efficient Jacobian updates.
    """
    x0 = np.array(x0, dtype=float)
    x_current = np.array(x0, dtype=float)
    
    # Homotopy function factory
    def make_F_t(t_val):
        def F_t(x_val):
            Fx = np.asarray(F(x_val), dtype=float)
            return t_val * Fx + (1.0 - t_val) * (x_val - x0)
        return F_t
    
    # Continue t from 0 to 1 with adaptive steps
    t = 0.0
    t_step = 0.2  # Start with larger steps to save calls
    max_t_steps = 25
    
    for t_iter in range(max_t_steps):
        t_new = min(t + t_step, 1.0)
        
        F_t = make_F_t(t_new)
        x_candidate = _broyden_2d(F_t, x_current, max_iter=30, tol=1e-12, eps=1e-6)
        
        # Check convergence
        F_candidate = F_t(x_candidate)
        if np.all(np.isfinite(F_candidate)) and np.linalg.norm(F_candidate) < 1e-10:
            x_current = x_candidate
            t = t_new
            if t >= 1.0:
                break
            # Increase step size if converged
            t_step = min(t_step * 1.4, 0.4)
        else:
            # Reduce step size and retry
            t_step *= 0.5
            if t_step < 1e-4:
                # Fine-grained continuation for remaining t
                remaining = 1.0 - t
                n_sub = 10
                for i in range(n_sub):
                    t_sub = t + remaining * (i + 1) / n_sub
                    if t_sub >= 1.0:
                        t_sub = 1.0
                    
                    F_t_sub = make_F_t(t_sub)
                    x_candidate = _broyden_2d(F_t_sub, x_current, max_iter=25, tol=1e-11, eps=1e-6)
                    
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
    
    # Fallback: direct Broyden from x_current
    x_direct = _broyden_2d(F, x_current, max_iter=40, tol=1e-13, eps=1e-6)
    F_direct = np.asarray(F(x_direct), dtype=float)
    if np.all(np.isfinite(F_direct)) and np.linalg.norm(F_direct) <= 1e-8:
        return x_direct
    
    # Fallback: direct Broyden from x0
    x_from_x0 = _broyden_2d(F, x0, max_iter=40, tol=1e-13, eps=1e-6)
    F_from_x0 = np.asarray(F(x_from_x0), dtype=float)
    if np.all(np.isfinite(F_from_x0)) and np.linalg.norm(F_from_x0) <= 1e-8:
        return x_from_x0
    
    # Return best candidate
    norm_direct = np.linalg.norm(F_direct) if np.all(np.isfinite(F_direct)) else np.inf
    norm_final = np.linalg.norm(F_final) if np.all(np.isfinite(F_final)) else np.inf
    norm_x0 = np.linalg.norm(F_from_x0) if np.all(np.isfinite(F_from_x0)) else np.inf
    
    if norm_direct <= norm_final and norm_direct <= norm_x0:
        return x_direct
    elif norm_final <= norm_x0:
        return x_current
    else:
        return x_from_x0
# EVOLVE-BLOCK-END
