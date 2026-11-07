"""Root finder for 3 fixed 2-D systems: Broyden's method with safeguards."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with x0 and use Broyden's method with secant-initialized Jacobian.
    2. Periodically recompute Jacobian via finite differences to prevent drift.
    3. If stuck, try a few random restarts from scaled versions of x0.
    4. Use damping if step is too large.
    
    Budget:
    - Time limit: ~115s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation that counts calls and handles errors
    call_count = [0]
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                call_count[0] += 1
                return None
            result = np.array(F(x_arr), dtype=float)
            call_count[0] += 1
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            call_count[0] += 1
            return None
    
    def compute_jacobian_fd(x, Fx, h=1e-7):
        """Compute Jacobian using forward finite differences. Costs 2 calls."""
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def broyden(x_start, max_iter=40, reanchor_every=5):
        """
        Broyden's method with periodic Jacobian re-anchoring.
        Returns (x, Fx, success) where success is True if ||Fx|| <= 1e-8.
        """
        x = np.array(x_start, dtype=float).copy()
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Initial Jacobian estimate using finite differences
        J = compute_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            # Try a different perturbation
            J = compute_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False
        
        det_J = np.linalg.det(J)
        if abs(det_J) < 1e-12:
            return x, Fx, False
        
        # Broyden's iterations with periodic re-anchoring
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= 480:
                break
            
            # Every reanchor_every iterations, recompute Jacobian via FD
            if reanchor_every > 0 and it > 0 and it % reanchor_every == 0:
                J_new = compute_jacobian_fd(x, Fx, h=1e-7)
                if J_new is not None:
                    det_new = np.linalg.det(J_new)
                    if abs(det_new) > 1e-12:
                        J = J_new
                else:
                    J_new = compute_jacobian_fd(x, Fx, h=1e-5)
                    if J_new is not None:
                        det_new = np.linalg.det(J_new)
                        if abs(det_new) > 1e-12:
                            J = J_new
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try with a slightly perturbed Jacobian
                J_pert = J + 1e-8 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_pert, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Damping: if step is too large, reduce it
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            
            x_new = x + dx
            
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Try a smaller step
                x_new = x + 0.5 * dx
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
            
            # Update Jacobian using Broyden's formula
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    # First attempt: Broyden from x0 with re-anchoring
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden(x0_arr, max_iter=40, reanchor_every=5)
    if success:
        return x
    
    # If first attempt failed, try restarts
    # Generate several restart points
    restarts = []
    
    # Scaled versions of x0
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(6):
        restarts.append(x0_arr + rng.randn(2) * np.linalg.norm(x0_arr) * 0.5)
    
    # Origin
    restarts.append(np.zeros(2))
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    for r in restarts:
        if not time_ok() or call_count[0] >= 450:
            break
        
        x_r, Fx_r, success_r = broyden(r, max_iter=25, reanchor_every=4)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    # Final check on best_x
    if Fx is not None and np.linalg.norm(Fx) <= 1e-8:
        return best_x
    
    return best_x
