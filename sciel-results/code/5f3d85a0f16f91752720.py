"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    # If already close, return
    if norm_F <= 1e-8:
        return x

    # Forward differences: 2 F calls per Jacobian (vs 4 for central)
    h = 1e-5
    max_iter = 80
    max_ls = 8
    deadline = start_time + 115  # Safety margin before 120s limit
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break

        # Compute Jacobian using forward differences (2 calls)
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            J[:, j] = (F(x + e) - Fx) / h
        
        # Try to solve J dx = -Fx (Newton step)
        try:
            dx = -np.linalg.solve(J, Fx)
        except np.linalg.LinAlgError:
            # Regularize if singular
            try:
                J_reg = J + 1e-6 * np.eye(2)
                dx = -np.linalg.solve(J_reg, Fx)
            except np.linalg.LinAlgError:
                break
        
        # Dampen large steps to improve convergence
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm
        
        # Line search with backtracking
        alpha = 1.0
        accepted = False
        
        for _ in range(max_ls):
            if time.time() > deadline:
                break
            x_candidate = x + alpha * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            
            if norm_candidate <= 1e-8:
                return x_candidate
            
            # Check for sufficient decrease
            if norm_candidate < norm_F * 0.5:
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
                accepted = True
                break
            else:
                alpha *= 0.5
        
        if not accepted:
            # Try a smaller step even without sufficient decrease
            x_candidate = x + 0.1 * dx
            F_candidate = F(x_candidate)
            norm_candidate = np.linalg.norm(F_candidate)
            if norm_candidate < norm_F:
                x = x_candidate
                Fx = F_candidate
                norm_F = norm_candidate
            else:
                break

    # Final polish: if we're close but not quite there, do a few more iterations
    if norm_F < 1e-4 and time.time() < deadline:
        for _ in range(15):
            if time.time() > deadline:
                break
            # Use smaller h for accuracy near root
            h_small = max(1e-8, h * 0.01)
            J = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_small
                J[:, j] = (F(x + e) - Fx) / h_small
            
            try:
                dx = -np.linalg.solve(J, Fx)
                x_candidate = x + dx
                F_candidate = F(x_candidate)
                norm_candidate = np.linalg.norm(F_candidate)
                if norm_candidate < norm_F:
                    x = x_candidate
                    Fx = F_candidate
                    norm_F = norm_candidate
                    if norm_F <= 1e-8:
                        return x
                else:
                    break
            except np.linalg.LinAlgError:
                break

    return x
# EVOLVE-BLOCK-END
