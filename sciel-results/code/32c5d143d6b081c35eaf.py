"""Root finder for 3 fixed 2-D systems: Broyden with adaptive re-anchoring and polishing."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method with adaptive re-anchoring (recompute Jacobian when needed).
    2. Damped Newton steps with backtracking line search.
    3. Multiple restarts from perturbed x0 if initial attempt fails.
    4. Final polishing phase with high-precision finite-difference Newton.
    
    Budget:
    - Time limit: ~110s internal (evaluator kills at 150s).
    - Call limit: < 480 calls per system.
    """
    start_time = time.time()
    time_limit = 110.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling and call counting."""
        try:
            result = np.array(F(x), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_J_forward(x, Fx, h=1e-7):
        """Compute Jacobian via forward finite differences (2 calls)."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_J_central(x, h=1e-6):
        """Compute Jacobian via central finite differences (5 calls total including F(x))."""
        Fx0 = safe_eval_F(x)
        if Fx0 is None:
            return None, None
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None, None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J, Fx0
    
    def broyden_solve(x_start, max_iter=40, reanchor_every=4):
        """
        Broyden's method with adaptive re-anchoring.
        Returns (x, Fx, success).
        """
        x = np.array(x_start, dtype=float)
        
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian via forward differences
        J = compute_J_forward(x, Fx, h=1e-7)
        if J is None:
            return x, Fx, False
        
        # Check invertibility
        try:
            det_J = np.linalg.det(J)
            if abs(det_J) < 1e-14:
                # Try larger h
                J = compute_J_forward(x, Fx, h=1e-5)
                if J is None:
                    return x, Fx, False
                det_J = np.linalg.det(J)
                if abs(det_J) < 1e-14:
                    return x, Fx, False
        except np.linalg.LinAlgError:
            return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 460:
                break
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Recompute Jacobian
                J_new, Fx_new = compute_J_central(x)
                if J_new is None:
                    break
                Fx = Fx_new
                if np.linalg.norm(Fx) <= 1e-8:
                    return x, Fx, True
                try:
                    dx = np.linalg.solve(J_new, -Fx)
                except np.linalg.LinAlgError:
                    break
                J = J_new
                continue
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 100.0:
                dx = dx * (100.0 / step_norm)
            
            # Backtracking line search
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Track best
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            # If step didn't reduce norm, try smaller steps
            if Fx_new_norm > np.linalg.norm(Fx) * 1.01:
                for alpha in [0.5, 0.25, 0.1, 0.05]:
                    x_trial = x + alpha * dx
                    Fx_trial = safe_eval_F(x_trial)
                    if Fx_trial is None:
                        continue
                    Fx_trial_norm = np.linalg.norm(Fx_trial)
                    if Fx_trial_norm < np.linalg.norm(Fx):
                        x_new = x_trial
                        Fx_new = Fx_trial
                        Fx_new_norm = Fx_trial_norm
                        if Fx_new_norm < best_norm:
                            best_x = x_new.copy()
                            best_Fx = Fx_new.copy()
                            best_norm = Fx_new_norm
                        if Fx_new_norm <= 1e-8:
                            return x_new, Fx_new, True
                        break
            
            # Update Jacobian
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            
            # Adaptive re-anchoring: recompute if it's time or if J is ill-conditioned
            need_reanchor = False
            if (it + 1) % reanchor_every == 0:
                need_reanchor = True
            else:
                # Check condition number periodically
                if it > 0 and it % 3 == 0:
                    try:
                        cond = np.linalg.cond(J)
                        if cond > 1e10:
                            need_reanchor = True
                    except np.linalg.LinAlgError:
                        need_reanchor = True
            
            if need_reanchor:
                J_new, Fx_reanchor = compute_J_central(x_new)
                if J_new is not None:
                    J = J_new
                    Fx = Fx_reanchor
                    if np.linalg.norm(Fx) < best_norm:
                        best_x = x_new.copy()
                        best_Fx = Fx.copy()
                        best_norm = np.linalg.norm(Fx)
                    if np.linalg.norm(Fx) <= 1e-8:
                        return x_new, Fx, True
                    x = x_new
                    continue
            else:
                if s_dot_s > 1e-16:
                    J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def polish(x_start, max_iter=5):
        """
        High-precision finite-difference Newton polishing.
        Uses central differences with small h for better accuracy.
        """
        x = np.array(x_start, dtype=float)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 475:
                break
            
            J, Fx = compute_J_central(x, h=1e-8)
            if J is None:
                break
            
            if np.linalg.norm(Fx) <= 1e-8:
                return x, Fx, True
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            if np.linalg.norm(Fx_new) <= 1e-8:
                return x_new, Fx_new, True
            
            # If step is tiny, we've converged as far as possible
            if np.linalg.norm(dx) < 1e-12:
                break
            
            x = x_new
        
        return x, Fx, np.linalg.norm(Fx) <= 1e-8 if Fx is not None else False
    
    # Main algorithm
    x0_arr = np.array(x0, dtype=float)
    
    # First attempt: Broyden from x0
    x, Fx, success = broyden_solve(x0_arr)
    if success:
        return x
    
    # If close, try polishing
    if Fx is not None and np.linalg.norm(Fx) < 1e-4:
        x_pol, Fx_pol, success_pol = polish(x)
        if success_pol:
            return x_pol
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Restart strategy
    restarts = []
    
    # Scaled versions
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5, 0.01, 10.0]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations
    rng = np.random.RandomState(123)
    scale_base = max(np.linalg.norm(x0_arr), 1.0)
    for _ in range(12):
        restarts.append(x0_arr + rng.randn(2) * scale_base * 0.3)
    
    # Origin and axis-aligned points
    restarts.append(np.zeros(2))
    restarts.append(np.array([1.0, 0.0]))
    restarts.append(np.array([0.0, 1.0]))
    restarts.append(np.array([-1.0, 0.0]))
    restarts.append(np.array([0.0, -1.0]))
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 430:
            break
        
        x_r, Fx_r, success_r = broyden_solve(r, max_iter=30, reanchor_every=4)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
                # Try polishing if close
                if norm_r < 1e-4:
                    x_pol, Fx_pol, success_pol = polish(x_r)
                    if success_pol:
                        return x_pol
                    if Fx_pol is not None and np.linalg.norm(Fx_pol) < best_norm:
                        best_norm = np.linalg.norm(Fx_pol)
                        best_x = x_pol.copy()
    
    # Final polish on best candidate
    if best_norm < 1e-4:
        x_pol, Fx_pol, success_pol = polish(best_x)
        if success_pol:
            return x_pol
        if Fx_pol is not None:
            return x_pol
    
    return best_x
