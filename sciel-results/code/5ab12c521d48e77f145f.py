"""Root finder for 3 fixed 2-D systems: Broyden's method with re-anchoring."""
import numpy as np
import time


def find_root_system(F, x0):
    """
    Find x in R^2 such that ||F(x)|| <= 1e-8.
    
    Strategy:
    1. Start with x0 and use Broyden's method with periodic Jacobian re-anchoring.
    2. If stuck, try several restarts from heuristic offsets.
    3. Use damping and backtracking if step is too large or no improvement.
    
    Budget:
    - Time limit: ~110s internal (evaluator kills at 150s).
    - Call limit: < 500 calls per system.
    """
    start_time = time.time()
    time_limit = 110.0
    
    def time_ok():
        return (time.time() - start_time) < time_limit
    
    def safe_eval_F(x):
        """Evaluate F with error handling."""
        try:
            result = F(np.array(x, dtype=float))
            result = np.array(result, dtype=float)
            if result.shape != (2,):
                result = result.reshape(2)
            if not np.all(np.isfinite(result)):
                return None
            return result
        except:
            return None
    
    def compute_jacobian(x, Fx, max_calls_remaining):
        """Compute Jacobian via finite differences (2 calls)."""
        calls = 0
        h = 1e-7
        J = np.zeros((2, 2))
        for j in range(2):
            if max_calls_remaining <= 0 or not time_ok():
                break
            e = np.zeros(2)
            e[j] = h
            Fx_e = safe_eval_F(x + e)
            if Fx_e is None:
                # Fallback: use identity column
                J[:, j] = np.eye(2)[:, j]
            else:
                J[:, j] = (Fx_e - Fx) / h
            calls += 1
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
        return J, calls
    
    def broyden(x0, max_iter=60, max_calls=480):
        """
        Broyden's method with periodic re-anchoring.
        Returns (x, Fx, num_calls, success).
        """
        x = np.array(x0, dtype=float)
        calls = 0
        
        # Initial evaluation
        Fx = safe_eval_F(x)
        if Fx is None:
            return x, None, calls, False
        calls += 1
        
        if np.linalg.norm(Fx) <= 1e-8:
            return x, Fx, calls, True
        
        # Initial Jacobian via finite differences
        J, j_calls = compute_jacobian(x, Fx, max_calls - calls)
        calls += j_calls
        
        if not np.all(np.isfinite(J)):
            J = np.eye(2)
        
        # Broyden's good method with periodic re-anchoring
        reanchor_every = 5  # Recompute Jacobian every 5 iterations
        for it in range(max_iter):
            if calls >= max_calls or not time_ok():
                break
            
            # Check convergence
            if np.linalg.norm(Fx) <= 1e-8:
                return x, Fx, calls, True
            
            # Periodic re-anchoring: recompute Jacobian
            if it > 0 and it % reanchor_every == 0:
                J, j_calls = compute_jacobian(x, Fx, max_calls - calls)
                calls += j_calls
                if not np.all(np.isfinite(J)):
                    J = np.eye(2)
            
            # Solve J * dx = -Fx
            try:
                dx = np.linalg.solve(J, -Fx)
            except np.linalg.LinAlgError:
                try:
                    dx = np.linalg.lstsq(J, -Fx, rcond=None)[0]
                except:
                    break
            
            # Damping: if step is too large, shrink it
            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm
            
            # Backtracking line search
            alpha = 1.0
            x_new = x + alpha * dx
            Fx_new = safe_eval_F(x_new)
            calls += 1
            
            # Backtrack if no improvement or invalid
            max_backtracks = 5
            for _ in range(max_backtracks):
                if Fx_new is None or np.linalg.norm(Fx_new) >= np.linalg.norm(Fx):
                    alpha *= 0.5
                    if alpha < 1e-10:
                        break
                    x_new = x + alpha * dx
                    Fx_new = safe_eval_F(x_new)
                    calls += 1
                else:
                    break
            
            if Fx_new is None or np.linalg.norm(Fx_new) >= np.linalg.norm(Fx):
                # Stuck, break
                break
            
            # Update Jacobian (Broyden's good method)
            s = x_new - x
            y = Fx_new - Fx
            s_norm_sq = s @ s
            if s_norm_sq > 1e-20:
                try:
                    J = J + np.outer(y - J @ s, s) / s_norm_sq
                except:
                    pass
            
            x = x_new
            Fx = Fx_new
        
        return x, Fx, calls, (Fx is not None and np.linalg.norm(Fx) <= 1e-8)
    
    # First try from x0
    x, Fx, calls, success = broyden(x0, max_iter=50, max_calls=480)
    if success:
        return x
    
    # If failed, try several restarts with different initial points
    best_x = x
    best_norm = np.linalg.norm(Fx) if Fx is not None else np.inf
    
    restarts = []
    x0_arr = np.array(x0, dtype=float)
    
    # Try x0 scaled by various factors
    for scale in [0.5, 2.0, 0.1, 5.0, -1.0]:
        restarts.append(x0_arr * scale)
    
    # Try small perturbations along each axis
    for axis in range(2):
        for delta in [-1.0, 1.0, -0.5, 0.5]:
            pt = x0_arr.copy()
            pt[axis] += delta
            restarts.append(pt)
    
    # Try origin and unit vectors
    restarts.append(np.array([0.0, 0.0]))
    restarts.append(np.array([1.0, 0.0]))
    restarts.append(np.array([0.0, 1.0]))
    restarts.append(np.array([1.0, 1.0]))
    
    for restart_pt in restarts:
        if not time_ok() or calls >= 490:
            break
        remaining = 490 - calls
        if remaining < 10:
            break
        x_r, Fx_r, calls_r, success_r = broyden(restart_pt, max_iter=30, max_calls=remaining)
        calls += calls_r
        if Fx_r is not None:
            norm_r = np.linalg.norm(Fx_r)
            if norm_r < best_norm:
                best_x = x_r
                best_norm = norm_r
            if success_r:
                return x_r
    
    return best_x
