"""Root finder for 3 fixed 2D systems using Broyden's method with line search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    x = np.array(x0, dtype=float)
    Fx = F(x)
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-8:
        return best_x
        
    # Initial Jacobian estimation using finite differences
    # Cost: 3 calls (1 already done, 2 more)
    h = 1e-6
    J = np.zeros((2, 2))
    
    # Column 0
    e0 = np.zeros(2)
    e0[0] = h
    Fx_e0 = F(x + e0)
    J[:, 0] = (Fx_e0 - Fx) / h
    
    # Column 1
    e1 = np.zeros(2)
    e1[1] = h
    Fx_e1 = F(x + e1)
    J[:, 1] = (Fx_e1 - Fx) / h
    
    # Broyden's method loop
    # Good-Broyden update: J_new = J + (y - J*s) * s^T / (s^T * s)
    # where s = x_new - x, y = F(x_new) - F(x)
    
    max_iter = 100
    tol = 1e-8
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check if we are close enough
        if best_norm <= tol:
            return best_x
            
        # Solve for Newton step
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # If J is singular, try a gradient descent step or re-estimate J
            # Gradient descent step: dx = -J^T Fx (approximate gradient of 0.5||F||^2)
            dx = -np.dot(J.T, Fx)
            # Normalize to prevent huge steps if J is ill-conditioned
            norm_dx = np.linalg.norm(dx)
            if norm_dx > 0:
                dx = dx / norm_dx * 0.1
            else:
                dx = np.array([1e-4, 0])
                
        # Line search
        alpha = 1.0
        f_norm_sq = np.dot(Fx, Fx)
        
        # Try up to 10 backtracks
        accepted = False
        for _ in range(10):
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            f_new_norm = np.linalg.norm(Fx_new)
            f_new_norm_sq = np.dot(Fx_new, Fx_new)
            
            # Armijo condition: sufficient decrease
            if f_new_norm_sq < f_norm_sq - 1e-4 * alpha * np.dot(dx, Fx):
                accepted = True
                break
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        if not accepted:
            # If line search fails, try a very small step in the steepest descent direction
            # or just accept the best we found so far and try to restart
            # For robustness, let's just take the smallest step we tried
            x_new = x + alpha * dx
            Fx_new = F(x_new)
            f_new_norm = np.linalg.norm(Fx_new)
            
            # If this is worse, we might be stuck. 
            # Let's re-estimate Jacobian if we haven't improved in a while?
            # Simple strategy: if no improvement, re-estimate J at current x
            if f_new_norm > best_norm:
                 # Re-estimate Jacobian at current x (costly but resets direction)
                 if time.time() > deadline - 5.0:
                     break
                 h = 1e-6
                 J = np.zeros((2, 2))
                 e0 = np.zeros(2); e0[0] = h
                 J[:, 0] = (F(x + e0) - Fx) / h
                 e1 = np.zeros(2); e1[1] = h
                 J[:, 1] = (F(x + e1) - Fx) / h
                 continue # Skip update, retry step with new J
        
        # Update Broyden's matrix
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = np.dot(s, s)
        if s_norm_sq > 1e-15:
            J = J + np.outer(y - J.dot(s), s) / s_norm_sq
            
        x = x_new
        Fx = Fx_new
        curr_norm = np.linalg.norm(Fx)
        
        if curr_norm < best_norm:
            best_norm = curr_norm
            best_x = x.copy()
            
        if best_norm <= tol:
            return best_x
            
    return best_x
# EVOLVE-BLOCK-END
