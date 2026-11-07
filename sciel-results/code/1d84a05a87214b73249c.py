"""Root finder for 3 fixed two-dimensional systems."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.array([0.0, 0.0])
        
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    if norm_F <= 1e-8:
        return x
    
    best_x = x.copy()
    best_norm = norm_F
    
    # Parameters
    h = 1e-5  # Step size for finite differences
    max_iter = 100
    max_alpha_halvings = 20
    
    # Compute initial Jacobian using forward differences
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = F(x + e)
        J[:, j] = (Fx_e - Fx) / h
        
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
            
        # Solve J dx = -Fx
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                dx = -np.linalg.solve(J + 1e-6 * np.eye(2), Fx)
            except np.linalg.LinAlgError:
                # Fallback to gradient step if J is totally bad
                # Approximate gradient of 0.5||F||^2 as J^T F
                grad = J.T @ Fx
                dx = -grad * 0.01
        
        # Limit step size to prevent divergence
        step_norm = np.linalg.norm(dx)
        if step_norm > 10.0:
            dx = dx / step_norm * 10.0
            
        # Backtracking Line Search
        alpha = 1.0
        accepted = False
        x_new = None
        Fx_new = None
        
        for _ in range(max_alpha_halvings):
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate
            
            if norm_candidate < best_norm:
                best_x = x_candidate.copy()
                best_norm = norm_candidate
                
            # Accept if norm decreases (sufficient decrease)
            if norm_candidate < norm_F:
                accepted = True
                x_new = x_candidate
                Fx_new = F_candidate
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # If no step was accepted, we are stuck or at a local minimum of the norm
            break
            
        # Update Jacobian (Broyden's Good)
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        
        if s_norm_sq > 1e-12:
            # J_new = J + (y - J*s) * s^T / (s^T * s)
            J += np.outer(y - J @ s, s) / s_norm_sq
            
        # Update state
        x = x_new
        Fx = Fx_new
        norm_F = np.linalg.norm(Fx)
        
        if norm_F <= 1e-8:
            return x
            
    return best_x
# EVOLVE-BLOCK-END
