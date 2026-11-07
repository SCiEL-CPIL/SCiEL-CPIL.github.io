import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin

    x = np.array(x0, dtype=float)
    Fx = F(x)
    norm_F = np.linalg.norm(Fx)
    
    # If already close, return
    if norm_F <= 1e-12:
        return x

    # Initial Jacobian estimation (Forward Difference)
    # h = 1e-6 is standard, but scale-aware is better
    scale = np.maximum(np.abs(x), 1.0)
    h = 1e-6 * scale
    
    J = np.zeros((2, 2))
    for j in range(2):
        e = np.zeros(2)
        e[j] = h[j]
        Fx_pert = F(x + e)
        J[:, j] = (Fx_pert - Fx) / h[j]
        
    # Broyden's Good Method
    # J_{k+1} = J_k + (y - J_k s) s^T / (s^T s)
    # where y = F(x_{k+1}) - F(x_k), s = x_{k+1} - x_k
    
    max_iter = 100
    best_x = x.copy()
    best_norm = norm_F

    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Solve J s = -Fx
        try:
            s = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Singular Jacobian, try pseudo-inverse or restart
            s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
        # Check step size, damp if too large
        step_norm = np.linalg.norm(s)
        if step_norm > 10.0:
            s = s / step_norm * 10.0

        x_new = x + s
        Fx_new = F(x_new)
        norm_F_new = np.linalg.norm(Fx_new)
        
        # Update best
        if norm_F_new < best_norm:
            best_norm = norm_F_new
            best_x = x_new.copy()
            
        if norm_F_new <= 1e-12:
            return x_new

        # Check for divergence
        if norm_F_new > 10 * norm_F:
            # Damp the step or restart Jacobian
            # Try half step
            x_half = x + 0.5 * s
            Fx_half = F(x_half)
            norm_F_half = np.linalg.norm(Fx_half)
            
            if norm_F_half < norm_F:
                x = x_half
                Fx = Fx_half
                norm_F = norm_F_half
                # Update Jacobian with this smaller step
                s_used = 0.5 * s
                y = Fx - Fx_prev # Wait, Fx is new Fx_half, Fx_prev was old Fx
                # Actually, let's just update J with the successful step
                y = Fx - Fx_old
                J = J + np.outer(y - J @ s_used, s_used) / (s_used @ s_used)
                x = x_half
                Fx = Fx_half
                norm_F = norm_F_half
                continue
            else:
                # Restart Jacobian at current x
                Fx = F(x) # Re-evaluate to be safe? No, we have Fx from before step? 
                # We have Fx (old) and Fx_new (bad). Let's just re-estimate J at x
                # This costs 3 calls.
                scale = np.maximum(np.abs(x), 1.0)
                h = 1e-6 * scale
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h[j]
                    Fx_pert = F(x + e)
                    J[:, j] = (Fx_pert - Fx) / h[j]
                continue

        # Successful step, update Broyden
        y = Fx_new - Fx
        s_used = x_new - x
        s_norm_sq = s_used @ s_used
        if s_norm_sq > 1e-12:
            J = J + np.outer(y - J @ s_used, s_used) / s_norm_sq
            
        x = x_new
        Fx = Fx_new
        norm_F = norm_F_new

    return best_x
