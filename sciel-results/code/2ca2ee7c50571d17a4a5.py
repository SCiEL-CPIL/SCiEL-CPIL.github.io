"""Root finder for 3 fixed two-dimensional systems using Broyden's method with restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy: Use Broyden's method (1 F call/iter after setup) with multiple restarts.
    Broyden updates Jacobian approximation, so each iteration costs 1 F(x) call.
    For 2D, initial Jacobian needs 3 calls (F(x0), F(x0+e1), F(x0+e2)).
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480  # Stay well under 500
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    def is_good(x):
        if x is None:
            return False
        x = np.asarray(x, dtype=float)
        if x.shape != (2,) or not np.all(np.isfinite(x)):
            return False
        fx = F_counted(x)
        fx = np.asarray(fx, dtype=float)
        if fx.shape != (2,) or not np.all(np.isfinite(fx)):
            return False
        return np.linalg.norm(fx) <= 1e-8
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            fx = F_counted(x)
            fx = np.asarray(fx, dtype=float)
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    def broyden(x_init, max_iter=30):
        """Broyden's method for 2D systems. Returns (x, success)."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-7
        
        # Initial Jacobian estimate: 3 calls
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, False
        n0 = np.linalg.norm(Fx0)
        if n0 <= 1e-10:
            update_best(x)
            return x, True
        
        # Estimate Jacobian
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return x, False
        
        J = np.column_stack([(Fx1 - Fx0) / h, (Fx2 - Fx0) / h])
        
        # Check if Jacobian is singular
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-15:
            # Perturb
            J += np.eye(2) * 1e-8
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Broyden step: solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
                step_norm = 5.0
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                dx = -dx * 0.5
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new)
            
            if n_new <= 1e-10:
                return x_new, True
            
            # Broyden update: J <- J + (Fx_new - Fx0 - J*dx) * (dx^T) / (dx^T * dx)
            s = x_new - x
            y = Fx_new - Fx0
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
            
            x = x_new
            Fx0 = Fx_new
            n0 = n_new
        
        return x, n0 <= 1e-10
    
    def polish(x_init, max_iter=10):
        """Final high-precision polish pass using exact Newton with finite-diff Jacobian."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-8
        
        # Get current F(x)
        Fx = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx)):
            return x, False
        n = np.linalg.norm(Fx)
        if n <= 1e-10:
            return x, True
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Compute exact finite-difference Jacobian (3 calls)
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            Fx1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx2 = np.asarray(F_counted(x + e2), dtype=float)
            
            if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
                break
            
            J = np.column_stack([(Fx1 - Fx) / h, (Fx2 - Fx) / h])
            
            detJ = np.linalg.det(J)
            if abs(detJ) < 1e-15:
                J += np.eye(2) * 1e-8
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
            
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx * (1.0 / step_norm)
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new)
            
            if n_new <= 1e-10:
                return x_new, True
            
            # Check for improvement
            if n_new > n * 1.01:
                # Step was too large, try half
                dx = dx * 0.5
                x_new = x + dx
                if not np.all(np.isfinite(x_new)):
                    break
                Fx_new = np.asarray(F_counted(x_new), dtype=float)
                if not np.all(np.isfinite(Fx_new)):
                    break
                n_new = np.linalg.norm(Fx_new)
                update_best(x_new)
                if n_new <= 1e-10:
                    return x_new, True
                if n_new > n:
                    break
            
            x = x_new
            Fx = Fx_new
            n = n_new
        
        return x, n <= 1e-10
    
    # Try Broyden from x0
    x_result, success = broyden(x0)
    if success and best_norm <= 1e-8:
        # Polish for higher precision
        x_polished, pol_success = polish(best_x, max_iter=5)
        if pol_success and best_norm <= 1e-8:
            return best_x
        return best_x
    
    # Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = [x0.copy()]
    
    # Perturbations at various scales and directions
    for s in [0.5, 1.0, 2.0, 0.1, 5.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            candidates.append(x0 + s * np.array([dx, dy]))
    
    # Also try scaled versions
    for factor in [0.5, 2.0]:
        candidates.append(x0 * factor)
    
    # Try Broyden from each candidate
    for x_init in candidates:
        if call_count >= MAX_CALLS or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        x_result, success = broyden(x_init)
        if success and best_norm <= 1e-8:
            # Polish for higher precision
            x_polished, pol_success = polish(best_x, max_iter=5)
            if pol_success and best_norm <= 1e-8:
                return best_x
            return best_x
    
    # If still not good enough, try a few more with smaller steps
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        # Try from best_x with Broyden
        x_result, success = broyden(best_x)
        if success and best_norm <= 1e-8:
            # Polish for higher precision
            x_polished, pol_success = polish(best_x, max_iter=5)
            if pol_success and best_norm <= 1e-8:
                return best_x
            return best_x
    
    # Final polish attempt if we have budget
    if best_norm > 1e-8 and call_count < MAX_CALLS and time_ok():
        x_polished, pol_success = polish(best_x, max_iter=5)
        if pol_success and best_norm <= 1e-8:
            return best_x
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
