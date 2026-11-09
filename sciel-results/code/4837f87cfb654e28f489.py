"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find a root x in R^2 of F(x) = 0 using derivative-free Levenberg-Marquardt
    with rank-1 Jacobian updates (Broyden) and adaptive damping.
    
    Strategy:
    1. Initialize Jacobian via finite differences (3 calls).
    2. LM iterations: solve (J^T J + lambda*I) dx = -J^T Fx, 1 F call/iter.
    3. Adapt lambda based on residual reduction.
    4. Broyden rank-1 update of J each iteration.
    5. Final high-precision polish if close but not converged.
    
    Budget: <500 calls to F. Time-guarded with 135s safety margin.
    """
    x = np.array(x0, dtype=float, copy=True)
    deadline = time.time() + 135.0
    call_count = 0
    best_x = x.copy()
    best_norm = np.inf

    def _call_F(x_val):
        nonlocal call_count
        if time.time() > deadline:
            return None
        call_count += 1
        result = F(x_val)
        result = np.asarray(result, dtype=float)
        return result

    # Initial evaluation
    Fx = _call_F(x)
    if Fx is None:
        return best_x if best_norm < np.inf else x
    norm_Fx = np.linalg.norm(Fx)
    if norm_Fx <= 1e-8:
        return x
    best_x = x.copy()
    best_norm = norm_Fx

    # Compute initial Jacobian via finite differences (3 calls total so far + 2 = 5)
    J = np.zeros((2, 2))
    h = 1e-7
    for j in range(2):
        e = np.zeros(2)
        e[j] = h
        Fx_e = _call_F(x + e)
        if Fx_e is None:
            # Fallback: use identity if we can't compute J
            J = np.eye(2)
            break
        J[:, j] = (Fx_e - Fx) / h

    # LM parameters
    lambda_lm = 1e-2
    lambda_min = 1e-12
    lambda_max = 1e12
    lambda_increase = 10.0
    lambda_decrease = 0.1
    max_lm_iters = 450  # Leave room for initial 5 calls + polish

    for it in range(max_lm_iters):
        if time.time() > deadline:
            break
        if call_count >= 490:  # Leave room for final checks
            break

        # Check current residual
        norm_Fx = np.linalg.norm(Fx)
        if norm_Fx <= 1e-8:
            return x
        if norm_Fx < best_norm:
            best_x = x.copy()
            best_norm = norm_Fx

        # Solve (J^T J + lambda*I) dx = -J^T Fx
        JtJ = J.T @ J
        JtFx = J.T @ Fx
        A = JtJ + lambda_lm * np.eye(2)
        b = -JtFx

        try:
            dx = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            try:
                dx = np.linalg.pinv(A) @ b
            except:
                break

        # Ensure finite step
        if not np.all(np.isfinite(dx)):
            lambda_lm *= lambda_increase
            continue

        x_new = x + dx

        # Check if x_new is finite
        if not np.all(np.isfinite(x_new)):
            lambda_lm *= lambda_increase
            if lambda_lm > lambda_max:
                break
            continue

        Fx_new = _call_F(x_new)
        if Fx_new is None:
            break
        norm_Fx_new = np.linalg.norm(Fx_new)

        if norm_Fx_new < best_norm:
            best_x = x_new.copy()
            best_norm = norm_Fx_new

        if norm_Fx_new <= 1e-8:
            return x_new

        # Adapt lambda based on residual reduction
        if norm_Fx_new < norm_Fx:
            # Accept step, decrease lambda (more Newton-like)
            lambda_lm = max(lambda_min, lambda_lm * lambda_decrease)
            
            # Broyden rank-1 update
            s = x_new - x
            y = Fx_new - Fx
            norm_s = np.linalg.norm(s)
            if norm_s > 1e-15:
                J = J + np.outer(y - J @ s, s.T) / (s.T @ s)
            
            x = x_new
            Fx = Fx_new
        else:
            # Reject step, increase lambda (more gradient descent)
            lambda_lm = min(lambda_max, lambda_lm * lambda_increase)
            if lambda_lm > lambda_max:
                break

    # --- Phase 2: High-precision polish pass ---
    # If we're close but not converged, try a few more iterations with tighter tolerance
    if best_norm < 1e-4 and call_count < 495:
        x = best_x.copy()
        Fx = _call_F(x)
        if Fx is not None:
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx <= 1e-8:
                return x
            
            # Recompute Jacobian for polish
            h_polish = 1e-8
            J_polish = np.zeros((2, 2))
            for j in range(2):
                e = np.zeros(2)
                e[j] = h_polish
                Fx_e = _call_F(x + e)
                if Fx_e is None:
                    break
                J_polish[:, j] = (Fx_e - Fx) / h_polish
            
            # A few Newton steps for final convergence
            for polish_it in range(10):
                if time.time() > deadline:
                    break
                if call_count >= 498:
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
                    dx = np.linalg.solve(J_polish, -Fx)
                    x_new = x + dx
                    if not np.all(np.isfinite(x_new)):
                        dx *= 0.1
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
                    
                    # Update Jacobian
                    s = x_new - x
                    y = Fx_new - Fx
                    if np.linalg.norm(s) > 1e-15:
                        J_polish = J_polish + np.outer(y - J_polish @ s, s.T) / (s.T @ s)
                    
                    x = x_new
                    Fx = Fx_new
                except np.linalg.LinAlgError:
                    break

    # Final check
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
