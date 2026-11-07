"""Root finder for 3 fixed 2-D systems: Broyden's method with global search."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Try Broyden's quasi-Newton method from x0.
    2. If that fails or converges slowly, try from multiple starting points.
    3. Use a time guard to stay within limits.
    """
    start_time = time.time()
    time_limit = 115.0  # Safety margin before 120s limit
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    def safe_norm(v):
        try:
            n = np.linalg.norm(v)
            return n if np.isfinite(n) else np.inf
        except:
            return np.inf
    
    def safe_eval_F(x):
        try:
            fx = F(x)
            fx = np.asarray(fx, dtype=float)
            if fx.shape != (2,) or not np.all(np.isfinite(fx)):
                return None
            return fx
        except:
            return None
    
    def broyden_from(x0_init, max_iter=450):
        """Broyden's method (good-Broyden) starting from x0_init."""
        x = np.array(x0_init, dtype=float)
        
        # First F evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return None, 1
        if safe_norm(Fx) <= 1e-12:
            return x, 1
        calls = 1
        
        # Initialize Jacobian approximation using finite differences
        # This costs 2 more calls
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2)
            e[j] = 1e-6
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                return None, calls + 1
            J[:, j] = (Fx_e - Fx) / 1e-6
            calls += 1
        
        if not time_ok():
            return x, calls
        
        # Broyden's iterations
        for _ in range(max_iter):
            if not time_ok():
                break
            if calls >= 495:
                break
            
            # Solve J dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                # Try pseudo-inverse
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Check step size
            step_norm = np.linalg.norm(dx)
            if not np.isfinite(step_norm) or step_norm < 1e-15:
                break
            
            # Damping: limit step to avoid divergence
            max_step = 1.0
            if step_norm > max_step:
                dx = dx * (max_step / step_norm)
            
            x_new = x + dx
            Fx_new = safe_eval_F(x_new)
            if Fx_new is None:
                break
            calls += 1
            
            # Check convergence
            n_new = safe_norm(Fx_new)
            if n_new <= 1e-12:
                return x_new, calls
            
            # Update Jacobian (Broyden's good rank-1 update)
            s = x_new - x
            y = Fx_new - Fx
            # J_new = J + (y - J*s) * s^T / (s^T * s)
            Js = J @ s
            diff = y - Js
            ss = s @ s
            if ss > 1e-30:
                J = J + np.outer(diff, s) / ss
            
            x = x_new
            Fx = Fx_new
        
        return x, calls
    
    # Phase 1: Try Broyden from x0
    x0_arr = np.array(x0, dtype=float)
    best_x = x0_arr.copy()
    best_norm = np.inf
    total_calls = 0
    
    result, calls = broyden_from(x0_arr)
    total_calls += calls
    if result is not None:
        fn = safe_norm(safe_eval_F(result) if safe_eval_F(result) is not None else np.array([np.inf, np.inf]))
        # Actually, we already know the norm from inside, but let's re-evaluate for safety
        # The function returns the point, and we trust it converged or is the best so far
        # Let's just track the best point by its residual
        # We need to be careful: broyden_from returns the last x, which may not be the best
        # Let's modify: track best inside or just return the final and check
        
    # Better approach: track best point and its residual during the search
    
    # Let me restructure: have broyden_from return (best_x, best_residual, calls)
    # For now, let's just try multiple starting points and keep the best
    
    candidates = [x0_arr]
    
    # Add some global search points if we have budget
    # Try points around x0 at various scales
    scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    for scale in scales:
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            if not time_ok():
                break
            offset = np.array([scale * np.cos(angle), scale * np.sin(angle)])
            candidates.append(x0_arr + offset)
    
    # Also try some random-ish points
    rng = np.random.RandomState(42)
    for _ in range(10):
        if not time_ok():
            break
        pt = x0_arr + rng.uniform(-10, 10, size=2)
        candidates.append(pt)
    
    # Sort candidates by distance to x0 (closest first) to prioritize
    # Actually, just try them in order, x0 first
    
    for cand in candidates:
        if not time_ok():
            break
        if total_calls >= 450:
            break
        
        result, calls = broyden_from(cand)
        total_calls += calls
        
        if result is not None:
            fx = safe_eval_F(result)
            if fx is not None:
                n = safe_norm(fx)
                if n < best_norm:
                    best_norm = n
                    best_x = result.copy()
                    if n <= 1e-8:
                        return best_x
    
    # If we found something good, return it
    if best_norm <= 1e-8:
        return best_x
    
    # Fallback: return the best we found
    if best_norm < np.inf:
        return best_x
    
    # Last resort: return x0
    return x0_arr
