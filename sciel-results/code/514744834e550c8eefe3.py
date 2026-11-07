"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x)=0 using a damped Newton method with
    adaptive finite-difference Jacobian, fallback to Broyden's method if
    Newton steps fail or are too large, and a final safeguarded line search.
    """
    x = np.array(x0, dtype=float)
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # If initial evaluation is bad, try a small perturbation
        x = x + 1e-3 * np.array([1.0, 0.0])
        Fx = F(x)
    
    if np.linalg.norm(Fx) <= 1e-12:
        return x
    
    # Use Broyden's method with rank-1 updates for robustness
    # Initialize Jacobian approximation
    eps = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = eps
        J[:, j] = (F(x + e) - Fx) / eps
    
    # Broyden's good method
    for _ in range(100):
        if np.linalg.norm(Fx) <= 1e-12:
            return x
        
        try:
            dx = np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, reset with finite differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = eps
                J[:, j] = (F(x + e) - Fx) / eps
            try:
                dx = np.linalg.solve(J, Fx)
            except np.linalg.LinAlgError:
                # Last resort: steepest descent on ||F||^2
                grad = J.T @ Fx
                step_size = 0.01
                x_new = x - step_size * grad
                Fx_new = F(x_new)
                if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    x = x_new
                    Fx = Fx_new
                    continue
                else:
                    break
        
        # Damping: if step is too large, shrink it
        step_norm = np.linalg.norm(dx)
        if step_norm > 10.0:
            dx = dx * (10.0 / step_norm)
        
        x_new = x - dx
        Fx_new = F(x_new)
        
        if not np.all(np.isfinite(Fx_new)):
            # Step led to invalid point, try smaller step
            for shrink in [0.5, 0.1, 0.01]:
                x_try = x - shrink * dx
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)) and np.linalg.norm(Fx_try) < np.linalg.norm(Fx):
                    x = x_try
                    Fx = Fx_try
                    break
            else:
                break
        
        # Check if step improved the objective
        if np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
            # Update Jacobian using Broyden's formula
            s = x_new - x
            y = Fx_new - Fx
            J = J + np.outer(y - J @ s, s) / (s @ s) if s @ s > 1e-15 else J
            x = x_new
            Fx = Fx_new
        else:
            # Line search: try smaller steps
            found = False
            for shrink in [0.5, 0.25, 0.1, 0.05, 0.01]:
                x_try = x - shrink * dx
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)) and np.linalg.norm(Fx_try) < np.linalg.norm(Fx):
                    s = x_try - x
                    y = Fx_try - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s) if s @ s > 1e-15 else J
                    x = x_try
                    Fx = Fx_try
                    found = True
                    break
            if not found:
                # No improvement, try perturbing
                x = x + 1e-4 * np.random.randn(2)
                Fx = F(x)
                # Reset Jacobian
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = eps
                    J[:, j] = (F(x + e) - Fx) / eps
    
    return x
# EVOLVE-BLOCK-END
