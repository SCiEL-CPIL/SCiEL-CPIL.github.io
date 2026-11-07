"""
Root finder for 3 fixed 2-D systems: Homotopy-guided Broyden with adaptive safeguards.

Strategy:
1. Use a homotopy path from the origin (where F is often well-behaved) to the
   target point x0, taking adaptive steps to stay in the basin of attraction.
2. At each homotopy stage, use Broyden's method with secant-initialized Jacobian
   and periodic re-anchoring to polish to high accuracy.
3. If the homotopy path fails, fall back to direct Broyden from x0 and restarts.
4. Strict call and time budget management.

Budget:
- Time limit: ~115s internal (evaluator kills at 150s).
- Call limit: < 500 calls per system.
"""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Homotopy path from origin to x0 with adaptive RK-style stepping.
    2. Broyden's method with secant-initialized Jacobian and periodic re-anchoring.
    3. Damped line search for robustness.
    4. Multiple diverse restarts if initial attempt fails.
    """
    start_time = time.time()
    time_limit = 115.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    # Safe F evaluation with call counting
    call_count = [0]
    call_limit = 490  # Stay under 500
    
    def safe_eval_F(x):
        """Evaluate F with error handling and call counting."""
        call_count[0] += 1
        try:
            x_arr = np.array(x, dtype=float)
            if x_arr.shape != (2,) or not np.all(np.isfinite(x_arr)):
                return None
            result = np.array(F(x_arr), dtype=float)
            if result.shape != (2,) or not np.all(np.isfinite(result)):
                return None
            return result
        except Exception:
            return None
    
    def compute_J_secant(x, Fx):
        """
        Compute Jacobian via 2-point secant (2 calls).
        Uses a perturbation in the direction of Fx (or x0 direction if Fx is small).
        """
        fx_norm = np.linalg.norm(Fx)
        if fx_norm > 1e-10:
            direction = -Fx / fx_norm
        else:
            direction = np.array([1.0, 0.0])
        
        h = 1e-5
        
        x_pert = x + h * direction
        Fx_pert = safe_eval_F(x_pert)
        if Fx_pert is None:
            return None
        
        perp = np.array([-direction[1], direction[0]])
        x_pert2 = x + h * perp
        Fx_pert2 = safe_eval_F(x_pert2)
        if Fx_pert2 is None:
            return None
        
        delta_x = h * np.column_stack([direction, perp])
        delta_f = np.column_stack([Fx_pert - Fx, Fx_pert2 - Fx])
        
        try:
            J = delta_f @ np.linalg.inv(delta_x)
        except np.linalg.LinAlgError:
            return None
        
        if not np.all(np.isfinite(J)):
            return None
            
        return J
    
    def compute_J_fd(x, Fx, h=1e-6):
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
    
    def broyden(x_start, max_iter=30, reanchor_every=4):
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
        
        # Initial Jacobian estimate using secant
        J = compute_J_secant(x, Fx)
        if J is None:
            J = compute_J_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False
        
        # Check invertibility
        try:
            np.linalg.solve(J, np.ones(2))
        except np.linalg.LinAlgError:
            J = compute_J_fd(x, Fx, h=1e-4)
            if J is None:
                return x, Fx, False
            try:
                np.linalg.solve(J, np.ones(2))
            except np.linalg.LinAlgError:
                return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_limit:
                break
            
            # Periodic Jacobian re-anchoring
            if reanchor_every > 0 and it > 0 and it % reanchor_every == 0:
                J_new = compute_J_fd(x, Fx, h=1e-6)
                if J_new is not None:
                    J = J_new
                else:
                    J_new = compute_J_fd(x, Fx, h=1e-5)
                    if J_new is not None:
                        J = J_new
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J_pert = J + 1e-8 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_pert, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            # Adaptive step damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 15.0:
                dx = dx * (15.0 / step_norm)
            
            x_new = x + dx
            
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                # Try a smaller step
                for alpha in [0.5, 0.25, 0.1]:
                    x_new = x + alpha * dx
                    Fx_new = safe_eval_F(x_new)
                    if Fx_new is not None:
                        break
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
    
    def homotopy_broyden(x0_arr, n_steps=10, max_iter_per_step=15):
        """
        Homotopy path from origin to x0 with Broyden polishing at each step.
        Returns (x, Fx, success).
        """
        # Start from origin
        x = np.zeros(2)
        
        # Evaluate at origin
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        # Homotopy path: t * x0 for t in [0, 1]
        # We'll take n_steps adaptive steps along this path
        target = x0_arr.copy()
        
        # If x0 is very close to origin, just use Broyden directly
        if np.linalg.norm(target) < 1e-10:
            return broyden(x, max_iter=max_iter_per_step)
        
        # Adaptive homotopy stepping
        t = 0.0
        for step in range(n_steps):
            if not time_ok() or call_count[0] >= call_limit - 50:
                break
            
            # Determine next t value
            # Use adaptive step size based on local behavior
            if step == 0:
                t_next = 0.1
            else:
                # Take larger steps as we get closer to target
                t_next = min(1.0, t + 0.1 + 0.05 * step)
            
            x_next = t_next * target
            
            # Try to move from current x to x_next using Broyden
            x_new, Fx_new, success = broyden(x_next, max_iter=max_iter_per_step)
            
            if Fx_new is not None:
                norm_new = np.linalg.norm(Fx_new)
                norm_old = np.linalg.norm(Fx) if Fx is not None else np.inf
                
                # Accept the step if we made progress
                if norm_new <= norm_old * 1.5 or norm_new <= 1e-4:
                    x = x_new
                    Fx = Fx_new
                    t = t_next
                    
                    if success:
                        return x, Fx, True
                else:
                    # Step was bad, try a smaller step
                    t_half = (t + t_next) / 2
                    x_half = t_half * target
                    x_half_res, Fx_half, success_half = broyden(x_half, max_iter=max_iter_per_step)
                    
                    if Fx_half is not None and np.linalg.norm(Fx_half) < norm_old:
                        x = x_half_res
                        Fx = Fx_half
                        t = t_half
                        if success_half:
                            return x, Fx, True
                    else:
                        # Give up on homotopy, return current best
                        break
            else:
                break
        
        return x, Fx, (Fx is not None and np.linalg.norm(Fx) <= 1e-8)
    
    # First attempt: Homotopy from origin to x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = homotopy_broyden(x0_arr, n_steps=8, max_iter_per_step=12)
    if success:
        return x
    
    # If homotopy failed, try direct Broyden from x0
    if not success:
        x_direct, Fx_direct, success_direct = broyden(x0_arr, max_iter=30, reanchor_every=4)
        if success_direct:
            return x_direct
        if Fx_direct is not None and (Fx is None or np.linalg.norm(Fx_direct) < np.linalg.norm(Fx)):
            x, Fx = x_direct, Fx_direct
    
    # If still failed, try restarts
    restarts = []
    
    # Scaled versions of x0
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0, -0.5, 0.2, 10.0]:
        restarts.append(scale * x0_arr)
    
    # Random perturbations
    rng = np.random.RandomState(42)
    for _ in range(8):
        restarts.append(x0_arr + rng.randn(2) * np.linalg.norm(x0_arr) * 0.5)
    
    # Origin and other fixed points
    restarts.append(np.zeros(2))
    restarts.append(np.array([1.0, 1.0]))
    restarts.append(np.array([-1.0, -1.0]))
    restarts.append(np.array([1.0, -1.0]))
    restarts.append(np.array([-1.0, 1.0]))
    
    best_x = x.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    for r in restarts:
        if not time_ok() or call_count[0] >= call_limit - 20:
            break
        
        x_r, Fx_r, success_r = broyden(r, max_iter=20, reanchor_every=3)
        if success_r:
            return x_r
        
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_norm = norm_r
                best_x = x_r.copy()
    
    return best_x
