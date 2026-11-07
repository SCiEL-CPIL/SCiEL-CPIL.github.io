"""Root finder for 3 fixed 2D systems using Secant-based Quasi-Newton (Powell's Method)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _safe_eval(F, x):
    """Safely evaluate F at x, returning (Fx, norm_Fx, is_valid)."""
    try:
        Fx = F(x)
        Fx = np.asarray(Fx, dtype=float)
        if Fx.shape != (2,):
            if Fx.size == 2:
                Fx = Fx.reshape(2)
            else:
                return None, float('inf'), False
        if not np.all(np.isfinite(Fx)):
            return None, float('inf'), False
        norm_Fx = float(np.linalg.norm(Fx))
        return Fx, norm_Fx, True
    except Exception:
        return None, float('inf'), False

def _powell_secant(F, x0, call_counter, max_calls, tol=1e-10, time_limit=145.0, start_time=None):
    """
    Powell's Secant Method for 2D systems.
    Maintains the inverse Jacobian approximation directly.
    Requires 2 initial calls to establish the secant direction.
    Subsequent iterations require 1 call each.
    """
    if start_time is None:
        start_time = time.time()
    
    x = np.array(x0, dtype=float)
    Fx, norm_F, valid = _safe_eval(F, x)
    call_counter[0] += 1
    
    if not valid:
        return x, float('inf'), call_counter
    
    if norm_F <= tol:
        return x, norm_F, call_counter

    # Initialize Inverse Jacobian A (2x2)
    # We need two points to define the secant.
    # Let x0 be the first point.
    # We need a second point x1.
    # Choose a small perturbation in the direction of -Fx if possible, or just axis-aligned.
    # To be robust, let's pick a step size based on the magnitude of Fx or just a small fixed step.
    # Standard Powell uses a specific initialization. For 2D, we can pick x1 = x0 + delta * e_i
    # where e_i is the unit vector in the direction of the largest component of Fx, or just (1,0) if Fx is small.
    
    # Let's try a step in the direction of -Fx to encourage immediate decrease.
    # If Fx is near zero, we are already done.
    # If Fx is large, a step in -Fx direction is good.
    # However, for the secant method to work well, the step shouldn't be too large or too small.
    # Let's use a step size of 1e-2 * max(1, ||Fx||) in the direction of -Fx normalized.
    
    if norm_F > 1e-12:
        dir_vec = -Fx / norm_F
    else:
        dir_vec = np.array([1.0, 0.0])
        
    step_size = 1e-2 * max(1.0, norm_F)
    
    x1 = x + step_size * dir_vec
    F1, norm_F1, valid1 = _safe_eval(F, x1)
    call_counter[0] += 1
    
    if not valid1:
        # If the second point is invalid, try a different direction
        x1 = x + np.array([step_size, 0.0])
        F1, norm_F1, valid1 = _safe_eval(F, x1)
        call_counter[0] += 1
        if not valid1:
             x1 = x + np.array([0.0, step_size])
             F1, norm_F1, valid1 = _safe_eval(F, x1)
             call_counter[0] += 1
             if not valid1:
                 return x, norm_F, call_counter

    # Now we have (x, Fx) and (x1, F1)
    # Secant equation: J * (x1 - x) = F1 - Fx
    # We want to maintain A = J^{-1}
    # A * (F1 - Fx) = x1 - x
    # Let s = x1 - x, y = F1 - Fx
    # We need to initialize A such that A * y = s.
    # A simple choice is to project s onto y.
    # A = outer(s, y) / (y^T y)
    # This is the "secant" initialization.
    
    s = x1 - x
    y = F1 - Fx
    
    denom = np.dot(y, y)
    if denom < 1e-12:
        # y is too small, points are too close or F is constant.
        # Fall back to identity inverse?
        A = np.eye(2)
    else:
        A = np.outer(s, y) / denom
        
    # Ensure A is finite
    if not np.all(np.isfinite(A)):
        A = np.eye(2)

    # Now iterate
    # Current best point is x1 if norm_F1 < norm_F, else x
    if norm_F1 < norm_F:
        x = x1
        Fx = F1
        norm_F = norm_F1
    else:
        # Keep x as the current point, but we have the pair (x, Fx) and (x1, F1)
        # The secant update formula assumes we are moving from the old point to the new point.
        # If we stay at x, we haven't taken a step.
        # Let's force a step using the current A.
        # Actually, the standard Powell method updates the point at every step.
        # If the second point was worse, we might have chosen a bad direction.
        # Let's restart the secant construction with a better second point if possible?
        # Or just proceed with the current A and hope the next step is good.
        # To be safe, let's re-initialize A using the pair (x, Fx) and a new point if the first pair was bad?
        # No, let's just proceed. The inverse Jacobian A is an approximation.
        
        # However, the standard update formula for A is:
        # A_new = A + (s - A*y) * y^T / (y^T * A * y)
        # This formula assumes we have taken a step from x_old to x_new = x_old + s,
        # resulting in F_new = F_old + y.
        # In our case, we have two points. We haven't "moved" yet in the iterative sense.
        # We treat (x, Fx) as the current state.
        # We will compute a step s_new = -A * Fx.
        # Then x_new = x + s_new.
        # Then F_new = F(x_new).
        # Then update A using s_new and y_new = F_new - Fx.
        
        pass # No specific action needed here, just proceed to loop.

    # Main Iteration
    for _ in range(400): # Max iterations per start
        if call_counter[0] >= max_calls - 10:
            break
        if time.time() - start_time > time_limit:
            break
            
        if norm_F <= tol:
            break
            
        # Compute step
        s = -A @ Fx
        if not np.all(np.isfinite(s)):
            break
            
        # Step size control / Line search
        # Powell's method often uses a fixed step or simple damping.
        # Let's try full step first, then backtrack if it increases norm significantly.
        alpha = 1.0
        x_new = None
        F_new = None
        norm_new = float('inf')
        
        # Backtracking line search
        for _ in range(10):
            x_trial = x + alpha * s
            F_trial, norm_trial, valid_trial = _safe_eval(F, x_trial)
            call_counter[0] += 1
            
            if not valid_trial:
                alpha *= 0.5
                continue
                
            # Accept if norm decreases
            if norm_trial < norm_F:
                x_new = x_trial
                F_new = F_trial
                norm_new = norm_trial
                break
            else:
                alpha *= 0.5
                
        if x_new is None:
            # No improvement found
            # Try a very small step? Or break?
            # If we can't decrease, we are likely stuck or at a local minimum of ||F||^2.
            break
            
        # Update Inverse Jacobian A
        # Secant condition: A_new * y = s_actual
        # where s_actual = x_new - x, y = F_new - Fx
        s_actual = x_new - x
        y_actual = F_new - Fx
        
        denom_secant = np.dot(y_actual, A @ y_actual)
        
        if abs(denom_secant) > 1e-12:
            # Broyden-like update for inverse Jacobian
            # A_new = A + (s_actual - A*y_actual) * y_actual^T / (y_actual^T * A * y_actual)
            term = (s_actual - A @ y_actual) / denom_secant
            A = A + np.outer(term, y_actual)
            
            if not np.all(np.isfinite(A)):
                # Reset A to identity or previous valid state?
                # Resetting to identity is a safe fallback
                A = np.eye(2)

        x = x_new
        Fx = F_new
        norm_F = norm_new

    return x, norm_F, call_counter

def find_root_system(F, x0):
    start_time = time.time()
    time_limit = 145.0
    
    x0_arr = np.array(x0, dtype=float)
    if not np.all(np.isfinite(x0_arr)):
        x0_arr = np.zeros(2)
        
    call_counter = [0]
    max_calls = 500
    
    best_x = x0_arr
    best_norm = float('inf')
    
    # Initial evaluation of x0
    F0, norm0, valid0 = _safe_eval(F, x0_arr)
    call_counter[0] += 1
    if valid0:
        best_norm = norm0
        best_x = x0_arr
    else:
        # If x0 is invalid, try origin
        x0_arr = np.zeros(2)
        F0, norm0, valid0 = _safe_eval(F, x0_arr)
        call_counter[0] += 1
        if valid0:
            best_norm = norm0
            best_x = x0_arr
        else:
            return np.zeros(2) # Fallback

    if best_norm <= 1e-8:
        return best_x

    # Generate starts
    starts = [x0_arr, np.zeros(2)]
    
    # Add points on a grid/circle
    radii = [0.5, 1.0, 2.0, 5.0, 10.0]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, 5*np.pi/4, 3*np.pi/2, 7*np.pi/4]
    for r in radii:
        for a in angles:
            starts.append(np.array([r * np.cos(a), r * np.sin(a)]))
            
    # Random points
    rng = np.random.RandomState(42)
    for _ in range(20):
        starts.append(rng.uniform(-20, 20, 2))
        
    for x_start in starts:
        if time.time() - start_time > time_limit:
            break
        if call_counter[0] >= max_calls - 5:
            break
            
        # Skip if we already have a good solution
        if best_norm <= 1e-8:
            break
            
        x_res, norm_res, _ = _powell_secant(F, x_start, call_counter, max_calls, time_limit=time_limit, start_time=start_time)
        
        if norm_res < best_norm:
            best_norm = norm_res
            best_x = x_res
            
        if best_norm <= 1e-8:
            break
            
    return best_x
# EVOLVE-BLOCK-END
