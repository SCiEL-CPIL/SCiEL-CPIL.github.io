"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Adaptive step size for finite differences
    h = 1e-5
    max_iter = 80
    max_ls = 10
    
    # Levenberg-Marquardt parameter
    lambda_lm = 1e-6
    
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        
        # Compute Jacobian using forward differences
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        
        # Try to solve J dx = -Fx with LM regularization
        dx = None
        lambda_try = lambda_lm
        for _lm_iter in range(8):
            try:
                J_reg = J + lambda_try * np.eye(2)
                dx = -np.linalg.solve(J_reg, Fx)
                break
            except np.linalg.LinAlgError:
                lambda_try *= 10
        
        if dx is None:
            break
        
        # Limit step size to prevent divergence
        dx_norm = np.linalg.norm(dx)
        if dx_norm > 10.0:
            dx = dx / dx_norm * 10.0
        
        # Line search with Armijo-like condition
        alpha = 1.0
        accepted = False
        x_new = x
        F_new = Fx
        norm_new = norm_F
        
        for _ in range(max_ls):
            if time.time() > deadline:
                break
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate
            
            # Accept if we get at least 50% improvement
            if norm_candidate < 0.5 * norm_F:
                accepted = True
                x_new = x_candidate
                F_new = F_candidate
                norm_new = norm_candidate
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # Try a smaller step or break
            # If we can't improve, try a much smaller step
            x_candidate = x + 0.01 * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate < norm_F:
                x_new = x_candidate
                F_new = F_candidate
                norm_new = norm_candidate
            else:
                # Stuck, try to adjust lambda
                lambda_lm *= 10
                if lambda_lm > 1e6:
                    break
                continue
        
        x = x_new
        Fx = F_new
        norm_F = norm_new
        
        # Track best
        if norm_F < best_norm:
            best_norm = norm_F
            best_x = x.copy()
        
        # Adaptive h: if norm is decreasing well, we can use smaller h
        if norm_F < 1e-4:
            h = max(1e-7, h * 0.5)
        
        # Reduce lambda if we're converging well
        if norm_F < 0.1 * best_norm or norm_F < 1e-6:
            lambda_lm = max(1e-10, lambda_lm * 0.1)
    
    # Return the best result found
    return best_x
# EVOLVE-BLOCK-END
