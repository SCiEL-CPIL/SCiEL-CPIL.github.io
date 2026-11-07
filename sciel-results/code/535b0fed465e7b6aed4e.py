"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search and a time guard.
    """
    x = np.array(x0, dtype=float).reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # If F(x0) is not finite, try a small perturbation
        for dx in [1e-3, 1e-2, 1e-1, 1.0, -1e-3, -1e-2, -1e-1, -1.0]:
            x_try = x + np.array([dx, 0])
            Fx_try = F(x_try)
            if np.all(np.isfinite(Fx_try)):
                x = x_try
                Fx = Fx_try
                break
        if not np.all(np.isfinite(Fx)):
            return x
    
    # Initialize Jacobian with finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    
    # Broyden's method
    for iteration in range(100):
        if time_up():
            break
        
        # Check convergence
        if np.linalg.norm(Fx) <= 1e-12:
            return x
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If singular, use pseudo-inverse
            dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
        
        # Adaptive line search
        alpha = 1.0
        Fx_norm = np.linalg.norm(Fx)
        success = False
        
        for _ in range(30):
            if time_up():
                break
            
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            
            if not np.all(np.isfinite(Fx_new)):
                alpha *= 0.5
                continue
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Armijo-like condition: sufficient decrease
            if Fx_new_norm < Fx_norm * (1 - 1e-4 * alpha):
                # Update Broyden
                s = x_new - x
                y = Fx_new - Fx
                J = J + np.outer(y - J @ s, s) / (s @ s)
                x = x_new
                Fx = Fx_new
                success = True
                break
            else:
                alpha *= 0.5
        
        if not success:
            # Try a different direction if line search failed
            # Use gradient descent step
            grad = J.T @ Fx
            dx = -grad / (np.linalg.norm(grad) + 1e-15)
            alpha = 0.1
            for _ in range(10):
                if time_up():
                    break
                x_new = x + alpha * dx
                Fx_new = F(x_new)
                if np.all(np.isfinite(Fx_new)) and np.linalg.norm(Fx_new) < np.linalg.norm(Fx):
                    s = x_new - x
                    y = Fx_new - Fx
                    J = J + np.outer(y - J @ s, s) / (s @ s)
                    x = x_new
                    Fx = Fx_new
                    break
                alpha *= 0.5
    
    return x
# EVOLVE-BLOCK-END
