"""Root finder for 3 fixed 2D systems using Broyden's Method with Backtracking Line Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    # Deadline guard: 150s limit, use 145s for safety
    start_time = time.time()
    time_limit = 145.0
    
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
        
    tol = 1e-8
    max_iter = 100
    max_restarts = 5
    fd_step = 1e-6
    
    best_x = x.copy()
    best_Fx = None
    best_norm = np.inf
    
    for restart in range(max_restarts):
        # If restarting, perturb the starting point slightly if we've been stuck
        if restart > 0:
            # Perturb from the best found so far or original x0
            base = best_x if best_norm < np.inf else x0
            x = base + 0.1 * (np.random.rand(2) - 0.5)
            if not np.all(np.isfinite(x)):
                x = base
        
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                continue
            
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x.copy()
                best_Fx = Fx.copy()
                
            if norm_Fx <= tol:
                return x
            
            # Compute initial Jacobian using finite differences
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = fd_step
                Fx_pert = F(x + e)
                Fx_pert = np.asarray(Fx_pert, dtype=float).reshape(2)
                J[:, j] = (Fx_pert - Fx) / fd_step
                
            # Broyden's Method Loop
            for i in range(max_iter):
                if time.time() - start_time > time_limit:
                    break
                
                # Solve for step direction
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    # If singular, use pseudo-inverse or restart
                    dx = np.linalg.pinv(J) @ (-Fx)
                
                # Backtracking Line Search
                alpha = 1.0
                accepted = False
                for _ in range(20): # Max 20 backtracking steps
                    x_new = x + alpha * dx
                    Fx_new = F(x_new)
                    Fx_new = np.asarray(Fx_new, dtype=float).reshape(2)
                    
                    if not np.all(np.isfinite(Fx_new)):
                        alpha *= 0.5
                        continue
                    
                    norm_Fx_new = np.linalg.norm(Fx_new)
                    
                    # Check if this is a better point
                    if norm_Fx_new < norm_Fx:
                        # Broyden Update
                        delta_x = alpha * dx
                        delta_F = Fx_new - Fx
                        
                        # Update Jacobian: J_new = J + (delta_F - J*delta_x) * delta_x^T / (delta_x^T * delta_x)
                        denom = np.dot(delta_x, delta_x)
                        if denom > 1e-12:
                            update_term = (delta_F - J @ delta_x) / denom
                            J += np.outer(update_term, delta_x)
                        
                        # Accept step
                        x = x_new
                        Fx = Fx_new
                        norm_Fx = norm_Fx_new
                        accepted = True
                        
                        if norm_Fx < best_norm:
                            best_norm = norm_Fx
                            best_x = x.copy()
                            best_Fx = Fx.copy()
                            
                        if norm_Fx <= tol:
                            return x
                        break
                    else:
                        alpha *= 0.5
                
                if not accepted:
                    # If no step was accepted, we are likely stuck or at a local minimum of the norm
                    break
                    
        except Exception:
            continue
            
    return best_x
# EVOLVE-BLOCK-END
