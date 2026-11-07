"""Root finder for 3 fixed 2D systems using homotopy/continuation method."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Finds x in R^2 such that ||F(x)|| <= 1e-8 using homotopy/continuation method.
    Tracks root path from trivial solution at t=0 to true root at t=1.
    """
    x = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x)):
        x = np.zeros(2)
    
    tol = 1e-8
    start_time = time.time()
    time_limit = 145.0

    def time_left():
        return time_limit - (time.time() - start_time)

    def F_norm(v):
        """Safely evaluate F and return (value, norm)."""
        try:
            fv = F(v)
            fv = np.asarray(fv, dtype=float)
            if fv.shape != (2,) or not np.all(np.isfinite(fv)):
                return None, np.inf
            return fv, np.linalg.norm(fv)
        except:
            return None, np.inf

    best_x = x.copy()
    best_norm = np.inf
    
    # Initial evaluation
    F_x0, best_norm = F_norm(x)
    if F_x0 is None:
        return np.zeros(2)
    
    if best_norm <= tol:
        return best_x

    # ===== HOMOTOPY / CONTINUATION METHOD =====
    # F_t(x) = t*F(x) + (1-t)*(x - x0)
    # At t=0: F_0(x) = x - x0, root is x0
    # At t=1: F_1(x) = F(x), root is what we want
    #
    # We track the path x(t) from t=0 to t=1
    # Using Newton's method at each t level with adaptive step size

    def F_t(xv, t):
        """Evaluate homotopy function F_t(x) = t*F(x) + (1-t)*(x - x0)"""
        Fv, _ = F_norm(xv)
        if Fv is None:
            return None
        return t * Fv + (1.0 - t) * (xv - x)

    def newton_step_homotopy(xv, t, max_iters=15):
        """
        Newton's method to find root of F_t near xv.
        Returns (x_new, F_new, norm_new, calls_used)
        """
        calls = 0
        x_local = xv.copy()
        
        F_local, norm_local = F_norm(x_local)
        calls += 1
        if F_local is None:
            return xv, None, np.inf, calls
        
        if norm_local <= tol:
            return x_local, F_local, norm_local, calls
        
        # Compute Jacobian of F_t via finite differences
        h = 1e-7
        scale = max(1.0, np.max(np.abs(x_local)))
        h = 1e-7 * scale
        
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = h
            Fx_e, _ = F_norm(x_local + e)
            if Fx_e is None:
                return x_local, F_local, norm_local, calls
            # Jacobian of F_t = t * J_F + (1-t) * I
            J[:, j] = t * (Fx_e - F_local) / h + (1.0 - t) * e[j]
            calls += 1
        
        # Newton iterations
        for _ in range(max_iters):
            if norm_local <= tol:
                return x_local, F_local, norm_local, calls
            
            det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            if abs(det) < 1e-14:
                # Add small regularization
                J[0,0] += 1e-10
                det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
                if abs(det) < 1e-14:
                    break
            
            try:
                s = np.linalg.solve(J, -F_local)
            except np.linalg.LinAlgError:
                break
            
            s_norm = np.linalg.norm(s)
            if s_norm < 1e-13:
                break
            
            # Line search
            alpha = 1.0
            f_prev = norm_local ** 2
            accepted = False
            
            for _ls in range(8):
                x_try = x_local + alpha * s
                F_try, norm_try = F_norm(x_try)
                calls += 1
                if F_try is None:
                    alpha *= 0.5
                    continue
                
                if norm_try <= tol:
                    return x_try, F_try, norm_try, calls
                
                if norm_try ** 2 < f_prev * 0.5 + 1e-14:
                    # Broyden update
                    delta_F = F_try - F_local - J @ s
                    s_sq = np.dot(s, s)
                    if s_sq > 1e-16:
                        J = J + np.outer(delta_F, s) / s_sq
                    
                    x_local = x_try
                    F_local = F_try
                    norm_local = norm_try
                    accepted = True
                    break
                else:
                    alpha *= 0.5
            
            if not accepted:
                break
        
        return x_local, F_local, norm_local, calls

    # Adaptive homotopy stepping
    t = 0.0
    x_current = x.copy()
    max_t_steps = 50
    min_dt = 0.02
    max_dt = 0.25
    
    for t_step in range(max_t_steps):
        if time_left() < 3.0:
            break
        
        # Choose adaptive dt
        dt = min(max_dt, max(min_dt, 0.1 * (1.0 - t)))
        t_new = min(1.0, t + dt)
        
        # Newton refinement at t_new starting from x_current
        x_new, F_new, norm_new, _calls = newton_step_homotopy(x_current, t_new)
        
        # Check if we converged at t=1
        if t_new >= 0.999 and F_new is not None:
            # Evaluate actual F (not F_t)
            F_actual, norm_actual = F_norm(x_new)
            if F_actual is not None:
                if norm_actual < best_norm:
                    best_norm = norm_actual
                    best_x = x_new.copy()
                if norm_actual <= tol:
                    return best_x
        
        # Check if we made progress (path tracking)
        if F_new is None or norm_new > 1e6:
            # Failed at this t level, try smaller step
            dt = max(min_dt, dt * 0.5)
            t_new = min(1.0, t + dt)
            x_new, F_new, norm_new, _calls = newton_step_homotopy(x_current, t_new)
            if F_new is None:
                # Try to recover: step back and try different path
                break
        
        x_current = x_new
        t = t_new
        
        if t >= 0.999:
            break
    
    # Final check at t=1
    F_final, norm_final = F_norm(x_current)
    if F_final is not None and norm_final < best_norm:
        best_norm = norm_final
        best_x = x_current.copy()
    
    if best_norm <= tol:
        return best_x

    # ===== FALLBACK: Multi-start Broyden if homotopy didn't converge =====
    if time_left() > 5.0:
        max_restarts = 15
        
        for restart in range(max_restarts):
            if time_left() < 2.0:
                break
            
            if restart == 0:
                x_local = x_current.copy()
                F_local, norm_local = F_norm(x_local)
            elif restart < 5:
                dirs = [np.array([1,0]), np.array([0,1]), np.array([1,1]), np.array([1,-1])]
                d = dirs[(restart-1) % 4]
                step = 0.1 * restart
                x_local = best_x + step * d
                F_local, norm_local = F_norm(x_local)
            else:
                if restart < 10:
                    radius = 0.5 * (restart - 4)
                    dir_vec = np.random.randn(2)
                    dir_vec /= np.linalg.norm(dir_vec)
                    x_local = best_x + radius * dir_vec
                else:
                    x_local = np.random.uniform(-10, 10, size=2)
                F_local, norm_local = F_norm(x_local)
            
            if F_local is None or not np.all(np.isfinite(x_local)):
                continue
            
            if norm_local < best_norm:
                best_norm = norm_local
                best_x = x_local.copy()
            
            if norm_local <= tol:
                return best_x
            
            # Broyden's method
            h = 1e-7
            scale = max(1.0, np.max(np.abs(x_local)))
            h = 1e-7 * scale
            
            J = np.zeros((2, 2))
            ok = True
            for j in range(2):
                e = np.zeros(2)
                e[j] = h
                Fx_e, _ = F_norm(x_local + e)
                if Fx_e is None:
                    ok = False
                    break
                J[:, j] = (Fx_e - F_local) / h
            
            if not ok:
                continue
            
            det = J[0,0]*J[1,1] - J[0,1]*J[1,0]
            if abs(det) < 1e-12:
                continue
            
            for it in range(25):
                if time_left() < 1.5:
                    break
                if norm_local <= tol:
                    return x_local
                
                try:
                    s = np.linalg.solve(J, -F_local)
                except np.linalg.LinAlgError:
                    break
                
                s_norm = np.linalg.norm(s)
                if s_norm < 1e-12:
                    break
                if s_norm > 10.0:
                    s = s * 10.0 / s_norm
                
                alpha = 1.0
                f_prev = norm_local ** 2
                accepted = False
                
                for _ls in range(8):
                    x_try = x_local + alpha * s
                    F_try, norm_try = F_norm(x_try)
                    if F_try is None:
                        alpha *= 0.5
                        continue
                    
                    if norm_try < best_norm:
                        best_norm = norm_try
                        best_x = x_try.copy()
                    
                    if norm_try <= tol:
                        return best_x
                    
                    if norm_try ** 2 < f_prev * 0.5 + 1e-12:
                        delta_F = F_try - F_local - J @ s
                        s_sq = np.dot(s, s)
                        if s_sq > 1e-16:
                            J = J + np.outer(delta_F, s) / s_sq
                        x_local = x_try
                        F_local = F_try
                        norm_local = norm_try
                        accepted = True
                        break
                    else:
                        alpha *= 0.5
                
                if not accepted:
                    break
    
    return best_x
# EVOLVE-BLOCK-END
