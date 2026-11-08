"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using Broyden's method with:
    1. Central-difference Jacobian initialization (5 calls)
    2. Broyden's good method with backtracking line search (1 call/iter)
    3. Final high-precision polish pass
    
    Budget: <500 calls to F. Initial: 5 calls. Broyden: ~1 call/iter.
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
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        return result
    
    # --- Phase 1: Initialize with central-difference Jacobian ---
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx
    
    # Central difference Jacobian for better initial accuracy
    J = np.zeros((2, 2))
    h = 1e-5
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_plus = _call_F(x + e)
        Fx_minus = _call_F(x - e)
        if Fx_plus is None or Fx_minus is None:
            # Fall back to forward difference
            Fx_e = _call_F(x + e)
            if Fx_e is not None:
                J[:, j] = (Fx_e - Fx) / h
            else:
                J[:, j] = 1.0  # identity fallback
        else:
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
    
    # If J is singular or has issues, use identity
    if not np.all(np.isfinite(J)) or np.linalg.cond(J) > 1e10:
        J = np.eye(2)
    
    # --- Phase 2: Broyden's method with line search ---
    for it in range(200):
        if time.time() > deadline:
            break
        if call_count >= 490:
            break
        
        # Solve J * dx = -Fx
        try:
            dx = np.linalg.solve(J, -Fx)
        except np.linalg.LinAlgError:
            try:
                dx = np.linalg.pinv(J) @ (-Fx)
            except:
                break
        
        # Line search with backtracking
        alpha = 1.0
        Fx_current = Fx.copy()
        norm_Fx_current = norm_Fx
        
        for ls_iter in range(10):
            x_trial = x + alpha * dx
            if not np.all(np.isfinite(x_trial)):
                alpha *= 0.5
                continue
            
            Fx_trial = _call_F(x_trial)
            if Fx_trial is None:
                break
            
            norm_Fx_trial = np.linalg.norm(Fx_trial)
            
            # Armijo condition: sufficient decrease
            if norm_Fx_trial <= best_norm:
                if norm_Fx_trial < best_norm:
                    best_x = x_trial.copy()
                    best_norm = norm_Fx_trial
                if norm_Fx_trial <= 1e-8:
                    return x_trial
                # Accept step
                s = x_trial - x
                y = Fx_trial - Fx
                if np.linalg.norm(s) > 1e-15:
                    # Broyden's good method update
                    J = J + np.outer(y - J @ s, s.T) / (s.T @ s)
                x = x_trial
                Fx = Fx_trial
                norm_Fx = norm_Fx_trial
                break
            else:
                alpha *= 0.5
        
        else:
            # No acceptable step found; try smaller step or break
            break
    
    # --- Phase 3: High-precision polish ---
    # If we have a good candidate, do a few more iterations with tighter tolerance
    if best_norm < 1e-4 and call_count < 495:
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx > 1e-8:
                # Recompute Jacobian with smaller h for better accuracy
                h = 1e-7
                J = np.zeros((2, 2))
                for j in range(2):
                    e = np.zeros(2)
                    e[j] = h
                    Fx_e = _call_F(x + e)
                    if Fx_e is not None:
                        J[:, j] = (Fx_e - Fx) / h
                    else:
                        break
                
                if np.all(np.isfinite(J)):
                    for polish_iter in range(5):
                        if call_count >= 498:
                            break
                        try:
                            dx = np.linalg.solve(J, -Fx)
                            x_new = x + dx
                            if np.all(np.isfinite(x_new)):
                                Fx_new = _call_F(x_new)
                                if Fx_new is not None:
                                    norm_Fx_new = np.linalg.norm(Fx_new)
                                    if norm_Fx_new <= 1e-8:
                                        return x_new
                                    if norm_Fx_new < best_norm:
                                        best_x = x_new.copy()
                                        best_norm = norm_Fx_new
                                    s = x_new - x
                                    y = Fx_new - Fx
                                    if np.linalg.norm(s) > 1e-15:
                                        J = J + np.outer(y - J @ s, s.T) / (s.T @ s)
                                    x = x_new
                                    Fx = Fx_new
                                    norm_Fx = norm_Fx_new
                        except:
                            break
    
    # Final check
    if best_norm < np.inf:
        Fx_final = _call_F(best_x)
        if Fx_final is not None:
            norm_final = np.linalg.norm(Fx_final)
            if norm_final <= 1e-8:
                return best_x
            if norm_final < best_norm:
                best_x = best_x.copy()
                best_norm = norm_final
    
    return best_x if best_norm < np.inf else x
# EVOLVE-BLOCK-END
