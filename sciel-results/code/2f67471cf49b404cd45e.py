# EVOLVE-BLOCK-START
import numpy as np


def find_root_system(F, x0):
    """
    Secant-Step Hybrid root finder for 2D systems.
    
    Strategy:
    1. Initialize Jacobian via forward finite differences (2 calls).
    2. Use Broyden's Good Method for quasi-Newton updates.
    3. Perform a simple backtracking line search to ensure global convergence.
    4. Re-anchor the Jacobian with finite differences every 4 iterations to correct drift.
    5. Monitor call count to stay under 500.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    # Helper to compute FD Jacobian
    def compute_jacobian(x_cur, Fx_cur):
        # Adaptive step size
        h = 1e-6 * max(1.0, np.linalg.norm(x_cur))
        J = np.zeros((2, 2))
        for i in range(2):
            e = np.zeros(2)
            e[i] = h
            Fx_pert = np.array(F(x_cur + e), dtype=float)
            J[:, i] = (Fx_pert - Fx_cur) / h
        return J

    # Initial function evaluation
    Fx = np.array(F(x), dtype=float)
    calls = 1
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx)
    
    if best_norm <= 1e-8:
        return x

    # Initialize Jacobian
    J = compute_jacobian(x, Fx)
    calls += 2  # 2 more calls for 2 columns

    max_iter = 100
    reanchor_freq = 4
    tol = 1e-10

    for i in range(max_iter):
        # Check call limit
        if calls > 450:
            break
            
        # Check convergence
        F_norm = np.linalg.norm(Fx)
        if F_norm <= tol:
            return x
        
        if F_norm < best_norm:
            best_norm = F_norm
            best_x = x.copy()

        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            except:
                # If solve fails, re-anchor
                if calls + 2 > 450: break
                J = compute_jacobian(x, Fx)
                calls += 2
                continue
        
        # Line search (Backtracking)
        alpha = 1.0
        max_ls_iter = 10
        F_new = None
        x_new = None
        
        for _ in range(max_ls_iter):
            if calls >= 450:
                break
            x_trial = x + alpha * dx
            F_trial = np.array(F(x_trial), dtype=float)
            calls += 1
            
            if not np.all(np.isfinite(F_trial)):
                alpha *= 0.5
                continue
                
            F_trial_norm = np.linalg.norm(F_trial)
            
            if F_trial_norm < F_norm:
                F_new = F_trial
                x_new = x_trial
                break
            else:
                alpha *= 0.5
                
        if F_new is None:
            # If line search failed, re-anchor and continue
            if calls + 2 > 450: break
            J = compute_jacobian(x, Fx)
            calls += 2
            continue

        # Prepare for Broyden update
        s = x_new - x
        y = F_new - Fx

        # Update x and Fx
        x = x_new
        Fx = F_new
        
        # Update best
        current_norm = np.linalg.norm(Fx)
        if current_norm < best_norm:
            best_norm = current_norm
            best_x = x.copy()
            
        # Check convergence again after line search
        if current_norm <= tol:
            return x

        # Broyden's Good Update
        s_dot_s = np.dot(s, s)
        if s_dot_s > 1e-15:
            J = J + np.outer(y - J @ s, s) / s_dot_s

        # Periodic re-anchoring
        if (i + 1) % reanchor_freq == 0:
            if calls + 2 > 450: break
            J = compute_jacobian(x, Fx)
            calls += 2

    return best_x
# EVOLVE-BLOCK-END
