# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    Uses Broyden's method with adaptive line search and time guard.
    Falls back to damped Newton if Broyden stagnates.
    """
    x = np.array(x0, dtype=float).copy().reshape(2)
    
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    def time_up():
        return time.time() > deadline
    
    def is_valid(x_val):
        return np.all(np.isfinite(x_val)) and x_val.shape == (2,)
    
    # Initial evaluation
    Fx = F(x)
    if not np.all(np.isfinite(Fx)):
        # Try a small perturbation
        for attempt in range(5):
            x_try = x + 0.1 * (attempt + 1) * np.array([1.0, 0.0])
            Fx_try = F(x_try)
            if np.all(np.isfinite(Fx_try)):
                x = x_try
                Fx = Fx_try
                break
        if not np.all(np.isfinite(Fx)):
            return x
    
    if np.linalg.norm(Fx) <= 1e-8:
        return x
    
    # Initialize Jacobian approximation with finite differences
    h = 1e-7
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        J[:, j] = (F(x + e) - Fx) / h
    
    # Broyden's method with line search
    best_x = x.copy()
    best_Fx = Fx.copy()
    best_norm = np.linalg.norm(Fx)
    
    for iteration in range(200):
        if time_up():
            break
        
        # Check convergence
        if best_norm <= 1e-8:
            return best_x
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try a small step in a random direction
            dx = -0.1 * Fx
            # Update J with a fresh finite difference
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = -0.1 * Fx
        
        # Ensure dx is finite
        if not np.all(np.isfinite(dx)):
            dx = -0.1 * Fx
        
        # Line search: try full step, then damped steps
        step_taken = False
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125]:
            x_try = x + alpha * dx
            if not is_valid(x_try):
                continue
            
            Fx_try = F(x_try)
            if not np.all(np.isfinite(Fx_try)):
                continue
            
            norm_try = np.linalg.norm(Fx_try)
            
            # Accept if it reduces the norm (Armijo-like)
            if norm_try < best_norm:
                # Update Broyden's approximation
                delta_x = x_try - x
                delta_F = Fx_try - Fx
                J = J + (delta_F - J @ delta_x) @ (delta_x.T) / (delta_x @ delta_x)
                
                x = x_try
                Fx = Fx_try
                best_x = x.copy()
                best_Fx = Fx.copy()
                best_norm = norm_try
                step_taken = True
                break
        
        if not step_taken:
            # Try a small step in the negative gradient direction
            dx_small = -0.01 * Fx
            x_try = x + dx_small
            if is_valid(x_try):
                Fx_try = F(x_try)
                if np.all(np.isfinite(Fx_try)):
                    norm_try = np.linalg.norm(Fx_try)
                    if norm_try < best_norm:
                        delta_x = dx_small
                        delta_F = Fx_try - Fx
                        J = J + (delta_F - J @ delta_x) @ (delta_x.T) / (delta_x @ delta_x)
                        x = x_try
                        Fx = Fx_try
                        best_x = x.copy()
                        best_Fx = Fx.copy()
                        best_norm = norm_try
                        continue
        
        # If no progress, try refreshing the Jacobian
        if not step_taken:
            h = 1e-7
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                J[:, j] = (F(x + e) - Fx) / h
    
    return best_x
# EVOLVE-BLOCK-END
