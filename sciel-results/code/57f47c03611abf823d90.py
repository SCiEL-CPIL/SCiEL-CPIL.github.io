"""Root finder for 3 fixed two-dimensional systems using Broyden's method with call optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Broyden's method (1 F call/iter after setup) with multiple restarts.
    2. Initial Jacobian via forward differences (3 calls).
    3. Polish phase with Newton (central differences) when close.
    4. Track calls carefully to stay under 500.
    5. Time-guarded with safety margin.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    def time_ok():
        return time.time() < deadline
    
    x0 = np.array(x0, dtype=float)
    call_count = 0
    MAX_CALLS = 480
    
    def F_counted(x):
        nonlocal call_count
        call_count += 1
        return F(x)
    
    best_x = x0.copy()
    best_norm = np.inf
    
    def update_best(x, fx=None):
        nonlocal best_x, best_norm
        try:
            x = np.asarray(x, dtype=float)
            if x.shape != (2,) or not np.all(np.isfinite(x)):
                return
            if fx is None:
                fx = F_counted(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return
            n = np.linalg.norm(fx)
            if np.isfinite(n) and n < best_norm:
                best_norm = n
                best_x = x.copy()
        except:
            pass
    
    def broyden(x_init, max_iter=25):
        """Broyden's method for 2D systems. Returns (x, fx, success)."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS or not time_ok():
            return x_init, None, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-7 * max(1.0, np.linalg.norm(x))
        
        # Initial F(x)
        Fx0 = np.asarray(F_counted(x), dtype=float)
        if not np.all(np.isfinite(Fx0)):
            return x, None, False
        n0 = np.linalg.norm(Fx0)
        if n0 <= 1e-10:
            update_best(x, Fx0)
            return x, Fx0, True
        
        # Forward difference Jacobian (2 more calls)
        e1 = np.array([h, 0.0])
        e2 = np.array([0.0, h])
        Fx1 = np.asarray(F_counted(x + e1), dtype=float)
        Fx2 = np.asarray(F_counted(x + e2), dtype=float)
        
        if not (np.all(np.isfinite(Fx1)) and np.all(np.isfinite(Fx2))):
            return x, Fx0, False
        
        J = np.column_stack([(Fx1 - Fx0) / h, (Fx2 - Fx0) / h])
        
        # Check singularity
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-12:
            J += np.eye(2) * 1e-8
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS or not time_ok():
                break
            
            # Solve J * dx = -Fx0
            try:
                dx = np.linalg.solve(J, -Fx0)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx0, rcond=None)[0]
                except:
                    break
            
            # Damping
            step_norm = np.linalg.norm(dx)
            if step_norm > 5.0:
                dx = dx * (5.0 / step_norm)
            
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
            update_best(x_new, Fx_new)
            
            if n_new <= 1e-10:
                return x_new, Fx_new, True
            
            # Broyden update
            s = x_new - x
            y = Fx_new - Fx0
            denom = np.dot(s, s)
            if denom > 1e-30:
                J = J + np.outer(y - J @ s, s) / denom
            
            x = x_new
            Fx0 = Fx_new
            n0 = n_new
        
        return x, Fx0, n0 <= 1e-10
    
    def polish(x_init, fx_init=None, max_iter=5):
        """Newton polish with central differences when close."""
        nonlocal call_count, best_norm, best_x
        if call_count >= MAX_CALLS - 10 or not time_ok():
            return x_init, fx_init, False
        
        x = np.array(x_init, dtype=float)
        h = 1e-8 * max(1.0, np.linalg.norm(x))
        
        if fx_init is None:
            Fx = np.asarray(F_counted(x), dtype=float)
        else:
            Fx = np.asarray(fx_init, dtype=float)
        
        n = np.linalg.norm(Fx)
        if not np.isfinite(n):
            return x, Fx, False
        
        for _ in range(max_iter):
            if call_count >= MAX_CALLS - 5 or not time_ok():
                break
            if n <= 1e-10:
                return x, Fx, True
            
            # Central difference Jacobian (4 calls)
            e1 = np.array([h, 0.0])
            e2 = np.array([0.0, h])
            Fx_p1 = np.asarray(F_counted(x + e1), dtype=float)
            Fx_m1 = np.asarray(F_counted(x - e1), dtype=float)
            Fx_p2 = np.asarray(F_counted(x + e2), dtype=float)
            Fx_m2 = np.asarray(F_counted(x - e2), dtype=float)
            
            if not all(np.all(np.isfinite(f)) for f in [Fx_p1, Fx_m1, Fx_p2, Fx_m2]):
                break
            
            J = np.column_stack([(Fx_p1 - Fx_m1) / (2*h), (Fx_p2 - Fx_m2) / (2*h)])
            
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            x_new = x + dx
            if not np.all(np.isfinite(x_new)):
                break
            
            Fx_new = np.asarray(F_counted(x_new), dtype=float)
            if not np.all(np.isfinite(Fx_new)):
                break
            
            n_new = np.linalg.norm(Fx_new)
            update_best(x_new, Fx_new)
            
            if n_new < n:
                x = x_new
                Fx = Fx_new
                n = n_new
            else:
                break
        
        return x, Fx, n <= 1e-10
    
    # Try Broyden from x0
    x_result, fx_result, success = broyden(x0)
    if success and best_norm <= 1e-8:
        return best_x
    
    # If close but not quite, try polish
    if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 15 and time_ok():
        x_pol, fx_pol, pol_success = polish(best_x, None)
        if pol_success and best_norm <= 1e-8:
            return best_x
    
    # Generate alternative starting points
    scale = max(1.0, np.linalg.norm(x0))
    candidates = []
    
    # Perturbations at various scales and directions
    for s in [0.5, 1.0, 2.0, 0.1, 5.0]:
        for dx, dy in [(1,0), (0,1), (1,1), (1,-1), (-1,0), (0,-1), (-1,-1), (-1,1)]:
            candidates.append(x0 + s * scale * np.array([dx, dy]) * 0.5)
    
    # Scaled versions
    for factor in [0.5, 2.0, 0.1]:
        candidates.append(x0 * factor)
    
    # Try Broyden from each candidate
    for x_init in candidates:
        if call_count >= MAX_CALLS or not time_ok():
            break
        if not np.all(np.isfinite(x_init)):
            continue
        x_result, fx_result, success = broyden(x_init)
        if success and best_norm <= 1e-8:
            return best_x
        
        # If close, try polish
        if best_norm < 1e-4 and best_norm > 1e-8 and call_count < MAX_CALLS - 15 and time_ok():
            x_pol, fx_pol, pol_success = polish(best_x, None)
            if pol_success and best_norm <= 1e-8:
                return best_x
    
    # Final polish attempt from best point
    if best_norm > 1e-8 and best_norm < 1.0 and call_count < MAX_CALLS - 15 and time_ok():
        x_pol, fx_pol, pol_success = polish(best_x, None)
        if pol_success and best_norm <= 1e-8:
            return best_x
    
    return best_x if np.all(np.isfinite(best_x)) else x0

# EVOLVE-BLOCK-END
