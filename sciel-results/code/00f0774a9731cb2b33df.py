import numpy as np
import time

def find_root_system(F, x0):
    """
    Find a root of F: R^2 -> R^2.
    Uses a hybrid strategy:
    1. Broyden's method with periodic FD Jacobian re-anchoring.
    2. If that fails, fallback to a few multi-start Broyden attempts.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin from 150s limit
    
    x0 = np.array(x0, dtype=float)
    
    call_count = 0
    def F_wrapper(x):
        nonlocal call_count
        call_count += 1
        try:
            return np.asarray(F(x), dtype=float)
        except Exception:
            return np.array([1e10, 1e10])

    def check_convergence(x, fx):
        return np.linalg.norm(fx) <= 1e-8

    def compute_jacobian_fd(x):
        """Compute Jacobian using forward differences (4 calls)."""
        h = 1e-5
        J = np.zeros((2, 2))
        f0 = F_wrapper(x)
        for i in range(2):
            xp = x.copy()
            xp[i] += h
            fp = F_wrapper(xp)
            J[:, i] = (fp - f0) / h
        return J

    def broyden_step(x, fx, J, step_type='broyden'):
        """
        Perform one Broyden step.
        Returns new x, new fx, updated J, and a flag if step was successful (didn't diverge).
        """
        # Solve J * dx = -fx
        try:
            dx = np.linalg.solve(J, -fx)
        except np.linalg.LinAlgError:
            return x, fx, J, False
        
        # Damping: if step is too large, damp it
        # Limit step to 1.0 initially? Or based on fx norm?
        # Let's limit step norm to 1.0 to avoid flying off to infinity
        step_norm = np.linalg.norm(dx)
        if step_norm > 1.0:
            dx = dx / step_norm * 1.0
        
        x_new = x + dx
        fx_new = F_wrapper(x_new)
        
        if not np.all(np.isfinite(fx_new)):
            return x, fx, J, False
            
        # Broyden update
        if step_type == 'broyden':
            dfx = fx_new - fx
            # Standard Broyden update: J_new = J + (dfx - J*dx) * dx^T / (dx^T * dx)
            denom = np.dot(dx, dx)
            if denom > 1e-12:
                J_new = J + np.outer(dfx - J @ dx, dx) / denom
            else:
                J_new = J
        else:
            J_new = J
            
        return x_new, fx_new, J_new, True

    # Initial check
    fx0 = F_wrapper(x0)
    if not np.all(np.isfinite(fx0)):
        # If x0 is invalid, try a small perturbation? 
        # But F(x0) failing might mean x0 is outside domain.
        # For now, return x0 as best guess, or try x0 + [1,1]?
        # Let's try a few simple points if x0 fails.
        # But the problem implies x0 is a valid starting point usually.
        # If F(x0) is not finite, it's likely a failure anyway.
        # Let's just try to proceed or return x0.
        # Actually, if F(x0) is not finite, we can't compute Jacobian.
        # Let's try x0 + [0.1, 0.1]
        x0 = x0 + np.array([0.1, 0.1])
        fx0 = F_wrapper(x0)
        if not np.all(np.isfinite(fx0)):
            return x0

    if check_convergence(x0, fx0):
        return x0

    # Initialize Jacobian
    J = compute_jacobian_fd(x0)
    
    x = x0
    fx = fx0
    
    max_iterations = 50
    reanchor_interval = 5 # Re-anchor every 5 steps
    
    for i in range(max_iterations):
        if time.time() > deadline - 2.0:
            break
            
        # Check convergence
        if check_convergence(x, fx):
            return x
            
        # Determine step type
        if i > 0 and i % reanchor_interval == 0:
            # Re-anchor: recompute Jacobian
            J = compute_jacobian_fd(x)
            # Note: compute_jacobian_fd uses 4 calls.
            # We should check if we are still under budget?
            # 50 iterations * (1 + 4 every 5) = 50 * 1.8 = 90 calls. Safe.
            
        # Perform step
        x_new, fx_new, J_new, success = broyden_step(x, fx, J)
        
        if not success:
            # Step failed (diverged or singular). 
            # Try a smaller step? Or re-anchor?
            # Let's try a smaller step by damping
            try:
                dx = np.linalg.solve(J, -fx)
                # Damp by 0.5
                dx = 0.5 * dx
                x_new = x + dx
                fx_new = F_wrapper(x_new)
                if np.all(np.isfinite(fx_new)):
                    # Update J with this damped step
                    dfx = fx_new - fx
                    denom = np.dot(dx, dx)
                    if denom > 1e-12:
                        J_new = J + np.outer(dfx - J @ dx, dx) / denom
                    else:
                        J_new = J
                    x, fx, J = x_new, fx_new, J_new
                    continue
                else:
                    # Still failed. Break and try multi-start?
                    break
            except:
                break
        else:
            x, fx, J = x_new, fx_new, J_new

    # If Broyden didn't converge, try a few multi-start points
    # Use random points in a small neighborhood
    best_x = x
    best_norm = np.linalg.norm(fx)
    
    random_state = np.random.RandomState(42)
    scales = [0.1, 0.5, 1.0]
    
    for scale in scales:
        if time.time() > deadline - 5.0:
            break
        for _ in range(3):
            offset = random_state.normal(0, scale, size=2)
            x_start = x0 + offset
            fx_start = F_wrapper(x_start)
            
            if not np.all(np.isfinite(fx_start)):
                continue
                
            if check_convergence(x_start, fx_start):
                return x_start
                
            # Run a short Broyden from this start
            J_start = compute_jacobian_fd(x_start)
            x_cur = x_start
            fx_cur = fx_start
            J_cur = J_start
            
            for _ in range(10): # Short run
                if time.time() > deadline - 5.0:
                    break
                if check_convergence(x_cur, fx_cur):
                    return x_cur
                x_cur, fx_cur, J_cur, success = broyden_step(x_cur, fx_cur, J_cur)
                if not success:
                    break
                if np.linalg.norm(fx_cur) < best_norm:
                    best_norm = np.linalg.norm(fx_cur)
                    best_x = x_cur
                    
    return best_x
