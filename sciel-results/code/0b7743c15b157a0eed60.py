"""Root finder for 3 fixed two-dimensional systems using Broyden's method with line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x of F: R^2 -> R^2 such that ||F(x)|| <= 1e-8.
    
    Uses Broyden's method (good update) with:
    - Finite-difference Jacobian for initialization
    - Adaptive step control (line search) for global convergence
    - Periodic Jacobian recomputation to prevent drift
    - Deadline guard to stay within time limits
    """
    x = np.array(x0, dtype=float)
    deadline = time.time() + 110  # 110s margin out of 120s limit
    
    # Initial function evaluation
    Fx = F(x)
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Compute initial Jacobian via finite differences
    def compute_jacobian(x, Fx):
        J = np.zeros((2, 2))
        h = 1e-7
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = F(x + e)
            J[:, j] = (Fx_plus - Fx) / h
        return J
    
    J = compute_jacobian(x, Fx)
    
    # Broyden's method with line search
    max_iter = 100
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    Fx_prev = Fx.copy()
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Check if we've found a root
        if best_norm <= 1e-8:
            return best_x
        
        # Compute Newton/Broyden step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, try adding regularization
            try:
                dx = np.linalg.solve(J + 1e-6 * np.eye(2), -Fx)
            except np.linalg.LinAlgError:
                # Give up on this iteration, try a small random step
                dx = 0.1 * np.random.randn(2)
        
        # Line search: find alpha in (0, 1] such that ||F(x + alpha*dx)|| < ||F(x)||
        alpha = 1.0
        Fx_norm_sq = np.dot(Fx, Fx)
        
        # Try full step first
        x_trial = x + alpha * dx
        Fx_trial = F(x_trial)
        Fx_trial_norm_sq = np.dot(Fx_trial, Fx_trial)
        
        # Backtracking line search
        if Fx_trial_norm_sq >= Fx_norm_sq:
            for _ in range(20):
                alpha *= 0.5
                x_trial = x + alpha * dx
                Fx_trial = F(x_trial)
                Fx_trial_norm_sq = np.dot(Fx_trial, Fx_trial)
                if Fx_trial_norm_sq < Fx_norm_sq:
                    break
                if time.time() > deadline:
                    break
            
            # If line search failed, use a very small step
            if Fx_trial_norm_sq >= Fx_norm_sq:
                alpha = 1e-4
                x_trial = x + alpha * dx
                Fx_trial = F(x_trial)
                Fx_trial_norm_sq = np.dot(Fx_trial, Fx_trial)
        
        # Compute s and y for Broyden update
        s = x_trial - x
        y = Fx_trial - Fx
        
        # Update x and Fx
        x = x_trial
        Fx = Fx_trial
        current_norm = np.linalg.norm(Fx)
        
        if current_norm < best_norm:
            best_x = x.copy()
            best_norm = current_norm
        
        # Check convergence
        if current_norm <= 1e-8:
            return x
        
        # Update Jacobian using Broyden's formula (good update)
        denom = np.dot(s, s)
        if denom > 1e-14:
            J = J + np.outer(y - J @ s, s) / denom
        
        # Periodically recompute Jacobian to prevent drift (every 5 iterations)
        if (it + 1) % 5 == 0:
            J = compute_jacobian(x, Fx)
        
        # Store Fx for next iteration's Broyden update
        Fx_prev = Fx.copy()
    
    # Return the best result found
    return best_x
# EVOLVE-BLOCK-END
