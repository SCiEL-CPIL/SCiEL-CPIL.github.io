# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2 using Newton's Method with Finite Difference Jacobian.
    """
    x0 = np.array(x0, dtype=float)
    
    def eval_F(x):
        try:
            Fx = F(x)
            Fx = np.array(Fx, dtype=float)
            if not np.all(np.isfinite(Fx)):
                return None, np.inf
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, np.inf

    def compute_jacobian_fd(x, Fx):
        """
        Compute Jacobian using forward differences.
        Reuses Fx for the base point.
        """
        h = 1e-5
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus, _ = eval_F(x + e)
            if Fx_plus is None:
                return None
            J[:, j] = (Fx_plus - Fx) / h
        return J

    def newton_step(x, Fx, norm_F):
        """
        Perform one Newton step with line search.
        Returns new x, new Fx, new norm, success_flag
        """
        J = compute_jacobian_fd(x, Fx)
        if J is None:
            return x, Fx, norm_F, False
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Fallback: Gradient descent step
            # Normalize F to avoid huge steps
            norm_F_val = np.linalg.norm(Fx)
            if norm_F_val > 1e-15:
                dx = -Fx / norm_F_val
            else:
                return x, Fx, norm_F, False

        # Line search
        alpha = 1.0
        x_new = x.copy()
        Fx_new = None
        norm_new = norm_F
        
        for _ in range(20):
            x_new = x + alpha * dx
            Fx_new, norm_new = eval_F(x_new)
            if Fx_new is None:
                alpha *= 0.5
                continue
            if norm_new < norm_F:
                break
            alpha *= 0.5
            
        if Fx_new is None or norm_new >= norm_F:
            # If line search fails, try a very small step or give up
            return x, Fx, norm_F, False

        return x_new, Fx_new, norm_new, True

    def run_newton(x_start, max_iter=50):
        x = x_start.copy()
        Fx, norm_F = eval_F(x)
        if Fx is None:
            return x, norm_F, False
        if norm_F <= 1e-8:
            return x, norm_F, True
            
        for _ in range(max_iter):
            x, Fx, norm_F, success = newton_step(x, Fx, norm_F)
            if not success:
                break
            if norm_F <= 1e-8:
                return x, norm_F, True
                
        return x, norm_F, norm_F <= 1e-8

    # Strategy: Try Newton from x0, then from random points if it fails.
    best_x = x0
    best_norm = np.inf

    # 1. Try from x0
    x_sol, norm_sol, success = run_newton(x0)
    if norm_sol < best_norm:
        best_x = x_sol
        best_norm = norm_sol
    if success:
        return x_sol

    # 2. If failed, try a few random restarts
    offsets = [
        [0.1, 0], [-0.1, 0], [0, 0.1], [0, -0.1],
        [1.0, 0], [-1.0, 0], [0, 1.0], [0, -1.0],
        [10.0, 0], [-10.0, 0], [0, 10.0], [0, -10.0],
        [0.1, 0.1], [-0.1, -0.1], [1.0, 1.0], [-1.0, -1.0],
        [0.01, 0], [-0.01, 0], [0, 0.01], [0, -0.01]
    ]
    
    for off in offsets:
        x_start = x0 + np.array(off)
        x_sol, norm_sol, success = run_newton(x_start)
        if norm_sol < best_norm:
            best_x = x_sol
            best_norm = norm_sol
        if success:
            return x_sol
            
    # 3. If all failed, return best point
    return best_x
# EVOLVE-BLOCK-END
