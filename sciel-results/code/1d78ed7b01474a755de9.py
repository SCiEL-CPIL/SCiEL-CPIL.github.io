"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def find_root_system(F, x0):
    # Robust root finder for 2D systems:
    # 1) Damped Newton with adaptive finite-difference Jacobian (central diff for accuracy).
    # 2) Levenberg-Marquardt fallback.
    # 3) Multi-start with diverse initial points.
    # 4) Time budget guard.
    
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s hard limit
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    x = np.array(x0, dtype=float).copy()
    n = 2
    
    def compute_F(val):
        return np.asarray(F(val), dtype=float)
    
    best_x = x.copy()
    best_norm = np.linalg.norm(compute_F(x))
    
    # Method 1: Damped Newton with central finite differences
    max_newton_iters = 100
    h_base = 1e-6
    
    def central_diff_jacobian(x_current, h):
        """Compute Jacobian using central differences (4 calls: 1 base + 4 perturbations)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            F_plus = compute_F(x_current + e)
            F_minus = compute_F(x_current - e)
            J[:, j] = (F_plus - F_minus) / (2 * h)
        return J
    
    def newton_from(x_start, max_iters=60):
        """Run damped Newton from x_start, return (x, norm_F)."""
        x_cur = x_start.copy()
        for it in range(max_iters):
            if not time_ok():
                break
            Fx = compute_F(x_cur)
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                nonlocal best_norm, best_x
                best_norm = norm_Fx
                best_x = x_cur.copy()
            if norm_Fx <= 1e-10:
                break
            
            # Adaptive step size for finite differences
            scale = max(1.0, np.linalg.norm(x_cur))
            h = h_base * scale ** 0.5
            J = central_diff_jacobian(x_cur, h)
            
            # Check if Jacobian is well-conditioned
            try:
                cond = np.linalg.cond(J)
            except np.linalg.LinAlgError:
                cond = np.inf
            
            if cond > 1e12:
                # Use LM step instead
                JtJ = J.T @ J
                lambda_lq = 1e-8
                dx = None
                for _ in range(30):
                    try:
                        dx = np.linalg.solve(JtJ + lambda_lq * np.eye(2), -J.T @ Fx)
                        break
                    except np.linalg.LinAlgError:
                        lambda_lq *= 10
                        if lambda_lq > 1e15:
                            break
                if dx is None:
                    break
            else:
                try:
                    dx = np.linalg.solve(J, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Damped step with backtracking line search
            step_size = 1.0
            x_new = x_cur + step_size * dx
            for _ in range(40):
                Fx_new = compute_F(x_new)
                norm_Fx_new = np.linalg.norm(Fx_new)
                if norm_Fx_new < norm_Fx:
                    break
                step_size *= 0.5
                x_new = x_cur + step_size * dx
                if step_size < 1e-16:
                    break
            x_cur = x_new
        
        Fx_final = compute_F(x_cur)
        norm_final = np.linalg.norm(Fx_final)
        if norm_final < best_norm:
            best_norm = norm_final
            best_x = x_cur.copy()
        return x_cur, norm_final
    
    # Run Newton from x0
    newton_from(x)
    
    # If converged, return early
    if best_norm <= 1e-8:
        if np.all(np.isfinite(best_x)):
            return best_x
    
    # Method 2: Levenberg-Marquardt from best point
    if time_ok() and best_norm > 1e-8:
        x_lm = best_x.copy()
        for it in range(100):
            if not time_ok():
                break
            Fx = compute_F(x_lm)
            norm_Fx = np.linalg.norm(Fx)
            if norm_Fx < best_norm:
                best_norm = norm_Fx
                best_x = x_lm.copy()
            if norm_Fx <= 1e-10:
                break
            
            scale = max(1.0, np.linalg.norm(x_lm))
            h = h_base * scale ** 0.5
            J = central_diff_jacobian(x_lm, h)
            
            JtJ = J.T @ J
            lambda_lq = 1e-8
            accepted = False
            for _ in range(50):
                try:
                    dx = np.linalg.solve(JtJ + lambda_lq * np.eye(2), -J.T @ Fx)
                except np.linalg.LinAlgError:
                    lambda_lq *= 10
                    continue
                x_trial = x_lm + dx
                Fx_trial = compute_F(x_trial)
                norm_trial = np.linalg.norm(Fx_trial)
                if norm_trial < norm_Fx:
                    lambda_lq *= 0.1
                    x_lm = x_trial
                    if norm_trial < best_norm:
                        best_norm = norm_trial
                        best_x = x_trial.copy()
                    accepted = True
                    break
                else:
                    lambda_lq *= 10
                if lambda_lq > 1e15:
                    break
            if not accepted:
                break
    
    if best_norm <= 1e-8:
        if np.all(np.isfinite(best_x)):
            return best_x
    
    # Method 3: Multi-start with diverse initial points
    if time_ok() and best_norm > 1e-8:
        # Generate diverse starting points
        starts = []
        # Small perturbations around best
        for scale in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
            for _ in range(3):
                starts.append(best_x + np.random.randn(2) * scale)
        # Points around origin
        for scale in [0.1, 1.0, 10.0]:
            for _ in range(2):
                starts.append(np.random.randn(2) * scale)
        # Points around x0
        for scale in [0.1, 1.0, 5.0]:
            for _ in range(2):
                starts.append(np.array(x0, dtype=float) + np.random.randn(2) * scale)
        
        for x_start in starts:
            if not time_ok():
                break
            if best_norm <= 1e-8:
                break
            newton_from(x_start, max_iters=50)
    
    # Final check: return best_x if it's finite, otherwise return x0
    if np.all(np.isfinite(best_x)):
        return best_x
    return np.array(x0, dtype=float)
# EVOLVE-BLOCK-END
