"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x such that ||F(x)|| <= 1e-8.
    Uses a robust damped Newton method with efficient Jacobian and adaptive stepping.
    """
    x = np.array(x0, dtype=float)
    tol = 1e-8
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Call counter to track budget
    call_count = [0]
    
    def F_counted(x):
        call_count[0] += 1
        return F(x)
    
    best_x = x.copy()
    best_norm = np.linalg.norm(F_counted(x))
    
    if best_norm <= tol:
        return best_x
    
    current_x = x.copy()
    max_restarts = 15
    
    for restart_iter in range(max_restarts):
        if time.time() > deadline - 5:
            break
        if call_count[0] > 450:  # Stay well under 500 call limit
            break
            
        x_local = current_x.copy()
        alpha = 1.0
        stagnation = 0
        
        for i in range(80):
            if time.time() > deadline - 1:
                return best_x
            if call_count[0] > 480:
                return best_x
            
            Fx = F_counted(x_local)
            norm_Fx = np.linalg.norm(Fx)
            
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_local.copy()
                stagnation = 0
            else:
                stagnation += 1
                
            if norm_Fx <= tol:
                return x_local
            
            if stagnation > 5:
                break
            
            # Compute Jacobian using forward differences (3 calls total: Fx already have, 2 more)
            J = np.zeros((2, 2))
            h = 1e-5 * max(1.0, np.linalg.norm(x_local))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_pert = F_counted(x_local + e)
                J[:, j] = (Fx_pert - Fx) / h
            
            # Newton step
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    JtJ = J.T @ J + 1e-6 * np.eye(2)
                    dx = np.linalg.solve(JtJ, -J.T @ Fx)
                except np.linalg.LinAlgError:
                    dx = np.random.randn(2) * 0.1
            
            # Adaptive step: try full step, then halve if needed (max 4 halvings)
            accepted = False
            for _ in range(4):
                x_try = x_local + alpha * dx
                F_try = F_counted(x_try)
                norm_try = np.linalg.norm(F_try)
                
                if norm_try < norm_Fx * 0.99:  # Require slight decrease
                    x_local = x_try
                    alpha = 1.0
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        if best_norm <= tol:
            return best_x
        
        # Restart strategy: try points around x0 and best_x with increasing radius
        if restart_iter < 3:
            current_x = x0 + np.random.uniform(-5, 5, size=2)
        else:
            radius = 0.5 * (restart_iter - 2)
            dir_vec = np.random.randn(2)
            dir_vec /= np.linalg.norm(dir_vec)
            current_x = best_x + radius * dir_vec

    return best_x
# EVOLVE-BLOCK-END
