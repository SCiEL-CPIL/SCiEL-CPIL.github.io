"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using a robust hybrid approach:
    1. Broyden's method (rank-1 update) as the primary solver.
    2. If Broyden stalls, try damped Newton with line search.
    3. Time-guarded: return best result before 140s wall-clock (safety margin).
    
    Budget: <500 calls to F. Broyden costs 1 call/iteration after initial setup (3 calls).
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 140.0  # safety margin
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf
    
    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None  # signal to stop
        if call_count >= 498:  # leave 2 calls for safety
            return None
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        if not np.all(np.isfinite(result)):
            return None
        return result
    
    # --- Initial evaluation ---
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx
    
    # Compute initial Jacobian using central differences for better accuracy
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        Fx_minus = _call_F(x - e)
        if Fx_plus is None or Fx_minus is None:
            # Fall back to forward difference
            Fx_plus = _call_F(x + e)
            if Fx_plus is None:
                return best_x if best_norm < np.inf else x
            J[:, j] = (Fx_plus - Fx) / h
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
    
    # Check if J is valid
    if not np.all(np.isfinite(J)) or np.linalg.norm(J) < 1e-15:
        # Use identity as fallback
        J = np.eye(2)
    
    # --- Phase 1: Broyden's method (rank-1) ---
    for it in range(200):
        if time.time() > deadline:
            break
        if call_count >= 495:
            break
        
        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except np.linalg.LinAlgError:
                dx = -Fx  # fallback: steepest descent
        
        # Damp if needed
        norm_dx = np.linalg.norm(dx)
        if norm_dx > 10.0:
            dx = dx / norm_dx * 10.0
        
        x_new = x + dx
        if not np.all(np.isfinite(x_new)):
            dx = dx * 0.1
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
        
        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        norm_Fx_new = np.linalg.norm(Fx_new)
        
        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new
        
        if norm_Fx_new <= 1e-8:
            return x_new
        
        # Broyden rank-1 update
        s = x_new - x
        y = Fx_new - Fx
        s_norm_sq = s.T @ s
        if s_norm_sq > 1e-15:
            J = J + np.outer(y - J @ s, s) / s_norm_sq
        
        x = x_new
        Fx = Fx_new
    
    # --- Phase 2: Damped Newton with line search ---
    # Recompute Jacobian for Newton
    J = np.zeros((2, 2))
    h = 1e-5
    Fx_ref = _call_F(x)
    if Fx_ref is None:
        return best_x if best_norm < np.inf else x
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _call_F(x + e)
        if Fx_e is None:
            return best_x if best_norm < np.inf else x
        J[:, j] = (Fx_e - Fx_ref) / h
    
    if not np.all(np.isfinite(J)) or np.linalg.norm(J) < 1e-15:
        J = np.eye(2)
    
    for it in range(100):
        if time.time() > deadline:
            break
        if call_count >= 495:
            break
        
        Fx = _call_F(x)
        if Fx is None:
            break
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx
        
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            dx = np.linalg.pinv(J) @ (-Fx)
        
        # Line search
        alpha = 1.0
        for ls in range(20):
            x_trial = x + alpha * dx
            if not np.all(np.isfinite(x_trial)):
                alpha *= 0.5
                continue
            Fx_trial = _call_F(x_trial)
            if Fx_trial is None:
                break
            norm_trial = np.linalg.norm(Fx_trial)
            if norm_trial < best_norm:
                best_x = x_trial.copy()
                best_norm = norm_trial
            if norm_trial <= 1e-8:
                return x_trial
            if norm_trial < norm_Fx * (1 - 1e-4 * alpha):
                # Accept step
                x = x_trial
                Fx = Fx_trial
                break
            alpha *= 0.5
        
        # Update Jacobian every 5 steps
        if (it + 1) % 5 == 0:
            J = np.zeros((2, 2))
            h = 1e-5
            Fx_ref = _call_F(x)
            if Fx_ref is None:
                break
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e = _call_F(x + e)
                if Fx_e is None:
                    break
                J[:, j] = (Fx_e - Fx_ref) / h
            if not np.all(np.isfinite(J)):
                J = np.eye(2)
    
    # --- Final check ---
    Fx_final = _call_F(x)
    if Fx_final is not None:
        norm_final = np.linalg.norm(Fx_final)
        if norm_final <= 1e-8:
            return x
        if norm_final < best_norm:
            best_x = x.copy()
            best_norm = norm_final
    
    return best_x
# EVOLVE-BLOCK-END
