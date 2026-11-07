"""Root finder for 3 fixed 2-D systems: homotopy path-following with adaptive RK4."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Try direct Broyden's method from x0 first (fast path).
    2. If that fails, use homotopy path-following:
       - Define H(x, t) = (1-t)*F(x) + t*(x - x0)
       - Start at t=1 where solution is x0
       - Integrate dx/dt from t=1 to t=0 using adaptive RK4
       - At t=0, H(x,0) = F(x), so we're at a root
    3. Polish with Newton's method if close.
    4. Multiple homotopy paths with different target points.
    """
    start_time = time.time()
    time_limit = 140.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    call_count = [0]
    
    def safe_eval_F(x):
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
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None
            J[:, j] = (Fx_e - Fx) / h
        return J
    
    def compute_jacobian_central(x, h=1e-6):
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_plus = safe_eval_F(x + e)
            Fx_minus = safe_eval_F(x - e)
            if Fx_plus is None or Fx_minus is None:
                return None
            J[:, j] = (Fx_plus - Fx_minus) / (2 * h)
        return J
    
    def newton_polish(x_start, Fx_start, max_iter=15, call_budget=480):
        x = np.array(x_start, dtype=float).copy()
        Fx = np.array(Fx_start, dtype=float).copy()
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break
            
            J = compute_jacobian_central(x, h=1e-6)
            if J is None:
                J = compute_jacobian_fd(x, Fx, h=1e-6)
                if J is None:
                    break
            
            if abs(np.linalg.det(J)) < 1e-12:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                break
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            x = x_new
            Fx = Fx_new
        
        return x, Fx, (np.linalg.norm(Fx) <= 1e-8) if Fx is not None else False
    
    def broyden_direct(x_start, max_iter=40, call_budget=450):
        x = np.array(x_start, dtype=float).copy()
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, False
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, True
        
        J = compute_jacobian_fd(x, Fx, h=1e-7)
        if J is None:
            J = compute_jacobian_fd(x, Fx, h=1e-5)
            if J is None:
                return x, Fx, False
        
        if abs(np.linalg.det(J)) < 1e-12:
            return x, Fx, False
        
        best_x = x.copy()
        best_Fx = Fx.copy()
        best_norm = np.linalg.norm(Fx)
        
        for it in range(max_iter):
            if not time_ok() or call_count[0] >= call_budget:
                break
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                J_pert = J + 1e-8 * np.eye(2)
                try:
                    dx = np.linalg.solve(J_pert, -Fx)
                except np.linalg.LinAlgError:
                    break
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 10.0:
                dx = dx * (10.0 / step_norm)
            
            alpha = 1.0
            found_step = False
            for _ in range(6):
                x_new = x + alpha * dx
                Fx_new = safe_eval_F(x_new)
                if Fx_new is None:
                    alpha *= 0.5
                    continue
                
                Fx_new_norm = np.linalg.norm(Fx_new)
                if Fx_new_norm < best_norm:
                    found_step = True
                    break
                alpha *= 0.5
            
            if not found_step:
                break
            
            x_new = x + alpha * dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            if Fx_new_norm < best_norm:
                best_x = x_new.copy()
                best_Fx = Fx_new.copy()
                best_norm = Fx_new_norm
            
            if Fx_new_norm <= 1e-8:
                return x_new, Fx_new, True
            
            s = x_new - x
            y = Fx_new - Fx
            s_dot_s = np.dot(s, s)
            if s_dot_s > 1e-16:
                J = J + np.outer(y - J @ s, s) / s_dot_s
            
            x = x_new
            Fx = Fx_new
        
        return best_x, best_Fx, best_norm <= 1e-8
    
    def homotopy_path_follow(x0_target, max_steps=200, dt_init=0.05, dt_min=1e-4, dt_max=0.5, call_budget=480):
        """
        Follow homotopy path H(x,t) = (1-t)*F(x) + t*(x - x0_target) = 0
        from t=1 (x=x0_target) to t=0 (F(x)=0).
        
        Uses adaptive RK4 on dx/dt = -[dH/dx]^{-1} * dH/dt
        where dH/dt = F(x) - (x - x0_target)
        and dH/dx = (1-t)*J_F + t*I
        """
        x = np.array(x0_target, dtype=float).copy()
        t = 1.0
        dt = dt_init
        
        for step in range(max_steps):
            if not time_ok() or call_count[0] >= call_budget:
                break
            if t <= 1e-10:
                break
            
            # Compute F(x) and Jacobian at current point
            Fx = safe_eval_F(x)
            if Fx is None:
                break
            
            # Check if we're already at a root
            Fx_norm = np.linalg.norm(Fx)
            if Fx_norm <= 1e-8:
                return x, Fx, True
            
            # Compute Jacobian of F
            J_F = compute_jacobian_fd(x, Fx, h=1e-7)
            if J_F is None:
                J_F = compute_jacobian_fd(x, Fx, h=1e-5)
                if J_F is None:
                    break
            
            # dH/dx = (1-t)*J_F + t*I
            dH_dx = (1.0 - t) * J_F + t * np.eye(2)
            
            # dH/dt = F(x) - (x - x0_target)
            dH_dt = Fx - (x - x0_target)
            
            # Solve dH_dx * dx/dt = -dH_dt
            try:
                dxdt = np.linalg.solve(dH_dx, -dH_dt)
            except np.linalg.LinAlgError:
                # Try with perturbation
                dH_dx_pert = dH_dx + 1e-8 * np.eye(2)
                try:
                    dxdt = np.linalg.solve(dH_dx_pert, -dH_dt)
                except np.linalg.LinAlgError:
                    break
            
            # Limit step size
            dxdt_norm = np.linalg.norm(dxdt)
            if dxdt_norm > 1.0:
                dxdt = dxdt * (1.0 / dxdt_norm)
            
            # RK4 integration
            def rk4_deriv(t_val, x_val):
                Fx_val = safe_eval_F(x_val)
                if Fx_val is None:
                    return None
                J_val = compute_jacobian_fd(x_val, Fx_val, h=1e-7)
                if J_val is None:
                    return None
                dH_dx_val = (1.0 - t_val) * J_val + t_val * np.eye(2)
                dH_dt_val = Fx_val - (x_val - x0_target)
                try:
                    return np.linalg.solve(dH_dx_val, -dH_dt_val)
                except np.linalg.LinAlgError:
                    return None
            
            # k1
            k1 = dxdt
            if k1 is None:
                break
            
            # k2
            x_k2 = x + 0.5 * dt * k1
            t_k2 = t + 0.5 * dt
            k2 = rk4_deriv(t_k2, x_k2)
            if k2 is None:
                break
            
            # k3
            x_k3 = x + 0.5 * dt * k2
            t_k3 = t + 0.5 * dt
            k3 = rk4_deriv(t_k3, x_k3)
            if k3 is None:
                break
            
            # k4
            x_k4 = x + dt * k3
            t_k4 = t + dt
            k4 = rk4_deriv(t_k4, x_k4)
            if k4 is None:
                break
            
            # Update
            x_new = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t_new = t - dt  # We're going from t=1 to t=0
            
            # Check convergence
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                dt *= 0.5
                continue
            
            Fx_new_norm = np.linalg.norm(Fx_new)
            
            # Adaptive step size
            if Fx_new_norm < 1e-6:
                dt = min(dt * 2.0, dt_max)
            elif Fx_new_norm > 10.0:
                dt *= 0.5
            else:
                dt = min(dt * 1.5, dt_max)
            
            dt = max(dt, dt_min)
            
            x = x_new
            t = t_new
            
            if Fx_new_norm <= 1e-8:
                return x, Fx_new, True
        
        # Return best point found
        Fx_final = safe_eval_F(x)
        if Fx_final is not None:
            return x, Fx_final, np.linalg.norm(Fx_final) <= 1e-8
        return x, None, False
    
    # First: try direct Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    x, Fx, success = broyden_direct(x0_arr, max_iter=40, call_budget=400)
    if success:
        return x
    
    best_x = x.copy() if Fx is not None else x0_arr.copy()
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    # Try Newton polish on best point
    if Fx is not None and best_norm < 1e-4:
        x_polish, Fx_polish, success_polish = newton_polish(best_x, Fx, max_iter=10, call_budget=450)
        if success_polish:
            return x_polish
        if Fx_polish is not None:
            norm_polish = np.linalg.norm(Fx_polish)
            if norm_polish < best_norm:
                best_norm = norm_polish
                best_x = x_polish.copy()
    
    # Try homotopy path-following with various target points
    n0_norm = np.linalg.norm(x0_arr)
    if n0_norm < 1e-10:
        n0_norm = 1.0
    
    targets = [x0_arr.copy()]
    
    # Add diverse targets
    for scale in [0.5, 2.0, -1.0, 0.2, 5.0]:
        targets.append(scale * x0_arr)
    
    # Perpendicular
    perp = np.array([-x0_arr[1], x0_arr[0]])
    perp_norm = np.linalg.norm(perp)
    if perp_norm > 1e-10:
        perp = perp / perp_norm * n0_norm
        targets.append(x0_arr + perp)
        targets.append(x0_arr - perp)
    
    # Origin
    targets.append(np.zeros(2))
    
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        targets.append(rng.randn(2) * max(1.0, n0_norm))
    
    for target in targets:
        if not time_ok() or call_count[0] >= 470:
            break
        
        x_h, Fx_h, success_h = homotopy_path_follow(target, max_steps=150, dt_init=0.1, dt_min=1e-4, dt_max=0.3, call_budget=470)
        if success_h:
            return x_h
        
        if Fx_h is not None:
            norm_h = np.linalg.norm(Fx_h)
            if norm_h < best_norm:
                best_norm = norm_h
                best_x = x_h.copy()
                
                # Try polishing
                if norm_h < 1e-4 and call_count[0] < 480:
                    x_polish, Fx_polish, success_polish = newton_polish(best_x, Fx_h, max_iter=8, call_budget=480)
                    if success_polish:
                        return x_polish
                    if Fx_polish is not None:
                        norm_polish = np.linalg.norm(Fx_polish)
                        if norm_polish < best_norm:
                            best_norm = norm_polish
                            best_x = x_polish.copy()
    
    # Final attempt: try a few more Broyden restarts
    for target in targets[:5]:
        if not time_ok() or call_count[0] >= 490:
            break
        x_b, Fx_b, success_b = broyden_direct(target, max_iter=25, call_budget=490)
        if success_b:
            return x_b
        if Fx_b is not None:
            norm_b = np.linalg.norm(Fx_b)
            if norm_b < best_norm:
                best_norm = norm_b
                best_x = x_b.copy()
    
    return best_x
