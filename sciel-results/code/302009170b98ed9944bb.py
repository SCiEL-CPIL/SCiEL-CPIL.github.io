"""Root finder for 3 fixed 2D systems using Broyden's method with multiple restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0  # Safety margin from 150s limit
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 490
    
    def safe_eval(x):
        if call_counter[0] >= max_calls:
            return None, float('inf')
        try:
            Fx = F(x)
            Fx = np.asarray(Fx, dtype=float)
            if Fx.shape != (2,):
                Fx = Fx.reshape(2)
            if not np.all(np.isfinite(Fx)):
                return None, float('inf')
            call_counter[0] += 1
            return Fx, np.linalg.norm(Fx)
        except Exception:
            return None, float('inf')

    def broyden_solve(x_start, max_iter=30, tol=1e-10):
        """
        Broyden's method for solving F(x) = 0.
        Returns (x, norm_F, calls_used)
        """
        x = x_start.copy()
        Fx, norm_F = safe_eval(x)
        if Fx is None:
            return x, float('inf'), 0
        if norm_F <= tol:
            return x, norm_F, 1
            
        # Initialize Jacobian approximation using finite differences
        h = 1e-5
        J = np.zeros((2, 2))
        for i in range(2):
            x_plus = x.copy()
            x_plus[i] += h
            F_plus, _ = safe_eval(x_plus)
            if F_plus is None:
                # Fallback: identity
                J = np.eye(2)
                break
            J[:, i] = (F_plus - Fx) / h
        else:
            if not np.all(np.isfinite(J)):
                J = np.eye(2)
                
        calls_used = 1 + 2  # 1 for Fx, 2 for J init (if both succeeded)
        
        for _ in range(max_iter):
            if time.time() - start_time > time_limit:
                break
            if call_counter[0] >= max_calls:
                break
                
            # Broyden step
            try:
                s = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    s = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    s = -Fx
                    
            if not np.all(np.isfinite(s)):
                break
                
            # Line search
            alpha = 1.0
            x_new = None
            Fx_new = None
            norm_new = float('inf')
            
            for _ in range(15):
                x_trial = x + alpha * s
                Fx_trial, norm_trial = safe_eval(x_trial)
                if Fx_trial is None:
                    alpha *= 0.5
                    continue
                    
                if norm_trial < (1 - 1e-4 * alpha) * norm_F:
                    x_new = x_trial
                    Fx_new = Fx_trial
                    norm_new = norm_trial
                    break
                else:
                    alpha *= 0.5
                    
            if x_new is None:
                break
                
            # Update Jacobian approximation
            delta_x = x_new - x
            delta_F = Fx_new - Fx
            denom = np.dot(delta_x, delta_x)
            
            if denom > 1e-12:
                J = J + np.outer(delta_F - J @ delta_x, delta_x) / denom
                
            x = x_new
            Fx = Fx_new
            norm_F = norm_new
            calls_used += 1 + 1  # 1 for trial, 1 for accept (approx)
            
            if norm_F <= tol:
                break
                
        return x, norm_F, calls_used

    # Multiple restart strategy
    # Generate several starting points
    start_points = [x0_arr.copy()]
    
    # Add some perturbed versions
    for scale in [0.1, 1.0, 10.0, 100.0]:
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                if dx == 0 and dy == 0:
                    continue
                perturbed = x0_arr.copy()
                perturbed[0] += dx * scale
                perturbed[1] += dy * scale
                start_points.append(perturbed)
                
    # Also try some random points in a reasonable range
    rng = np.random.default_rng(42)
    for _ in range(5):
        random_point = rng.uniform(-10, 10, size=2)
        start_points.append(random_point)
        
    # Try each starting point with Broyden's method
    best_x = x0_arr.copy()
    best_norm = float('inf')
    
    for x_start in start_points:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls:
            break
            
        x_result, norm_result, _ = broyden_solve(x_start, max_iter=20)
        
        if norm_result < best_norm:
            best_norm = norm_result
            best_x = x_result.copy()
            
        if best_norm <= 1e-8:
            break
            
    # Final polish if we have a good candidate
    if best_norm < 1e-4:
        for _ in range(3):
            if call_counter[0] >= max_calls:
                break
            Fx, norm_F = safe_eval(best_x)
            if Fx is None:
                break
            if norm_F <= 1e-8:
                break
                
            # Finite difference Jacobian for polish
            h = 1e-6
            J_fd = np.zeros((2, 2))
            for i in range(2):
                x_plus = best_x.copy()
                x_plus[i] += h
                F_plus, _ = safe_eval(x_plus)
                if F_plus is None:
                    J_fd = None
                    break
                J_fd[:, i] = (F_plus - Fx) / h
                
            if J_fd is None:
                break
                
            try:
                s = np.linalg.solve(J_fd, -Fx)
                best_x = best_x + s
            except:
                break
                
    return best_x
# EVOLVE-BLOCK-END
