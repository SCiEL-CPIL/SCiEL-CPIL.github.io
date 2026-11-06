"""Root finding for ill-conditioned problems using Trust Region Reflection."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a Trust Region method with reflection.
    This approach models f locally as a quadratic and takes steps within a 
    trust region to ensure robustness against ill-conditioning.
    """
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill
    TOL = 1e-8
    MAX_EVALS = 500

    eval_count = 0

    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result

    def check_time():
        return (time.time() - start_time) < DEADLINE

    # Initialize
    try:
        x = float(x0)
        fx = call_f(x)
    except:
        return x0

    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx) if math.isfinite(fx) else float('inf')

    if not math.isfinite(fx):
        # Try to find a finite point
        for dx in [1e-6, -1e-6, 1e-4, -1e-4, 1e-2, -1e-2, 1.0, -1.0]:
            try:
                x_try = x + dx
                fx_try = call_f(x_try)
                if math.isfinite(fx_try):
                    x = x_try
                    fx = fx_try
                    best_x = x
                    best_fx = fx
                    best_abs_fx = abs(fx)
                    break
            except:
                pass
        
        if not math.isfinite(fx):
            return best_x

    if best_abs_fx <= TOL:
        return best_x

    # Trust Region Parameters
    # Initial trust radius: based on scale of x
    delta = max(1e-4, abs(x) * 0.1)
    if not math.isfinite(delta) or delta == 0:
        delta = 1e-4
    
    # Model parameters
    # We will approximate f(x) ~ a + b*(x-x0) + c*(x-x0)^2
    # But simpler: use Secant-like step within trust region?
    # Let's use a standard Trust Region for 1D:
    # m(s) = f(x) + f'(x)s + 0.5 * f''(x) s^2
    # We need derivatives. Numerical derivatives are expensive (2 evals).
    # Alternative: Use "Trust Region" with a linear model? That's just Newton/Secant bounded.
    # Let's use a robust hybrid: 
    # 1. Estimate derivative via central difference if possible, or forward.
    # 2. Propose Newton step.
    # 3. If step > delta, clamp to boundary.
    # 4. Evaluate. 
    # 5. If actual reduction >= 0.25 * predicted reduction, accept and increase delta.
    #    Else, reject and decrease delta.
    # This is robust to ill-conditioning because it prevents large divergent steps.

    # State
    current_x = x
    current_fx = fx
    current_delta = delta
    
    # We need a derivative estimate.
    # Strategy:
    # - If we have a previous point, use secant slope.
    # - Else, use forward difference.
    
    # To save evals, we try to reuse points.
    # However, simple state:
    # x_prev, fx_prev
    # Initially, we need 2 points to start secant.
    
    # Let's initialize with a small step to get a second point for secant slope
    h = max(1e-8, abs(current_x) * 1e-6)
    x2 = current_x + h
    try:
        fx2 = call_f(x2)
    except:
        return best_x
    
    if not math.isfinite(fx2):
        # Try other side
        x2 = current_x - h
        try:
            fx2 = call_f(x2)
        except:
            return best_x
        if not math.isfinite(fx2):
            return best_x
        
        # Update best
        if abs(fx2) < best_abs_fx:
            best_x = x2
            best_fx = fx2
            best_abs_fx = abs(fx2)
        
        if best_abs_fx <= TOL:
            return best_x

    # Now we have (current_x, current_fx) and (x2, fx2)
    # Secant slope
    slope = (fx2 - current_fx) / (x2 - current_x)
    
    # Trust Region Loop
    # Variables:
    # x, fx : current point
    # x_prev, fx_prev : previous point (for secant)
    # delta : trust radius
    # We maintain two points: x and x_prev.
    
    x_curr = current_x
    fx_curr = current_fx
    x_prev = x2
    fx_prev = fx2
    
    # Ensure x_curr is the one with smaller |f| if possible? 
    # Usually trust region stays at the current iterate.
    # Let's stick to x_curr as the center.
    
    # If fx_curr is not the best, maybe we should center there?
    # For now, center at x_curr.
    
    # Initial delta
    delta = max(1e-4, abs(x_curr) * 0.1)
    
    # If the distance between x_curr and x_prev is larger than delta, 
    # the secant step might be outside. That's fine, we clamp.
    
    for i in range(150): # Max iterations, limited by evals anyway
        if not check_time() or eval_count >= MAX_EVALS:
            break
        
        if abs(fx_curr) <= TOL:
            return x_curr
        
        # 1. Estimate Derivative
        # Use secant slope if points are distinct
        if abs(x_curr - x_prev) > 1e-15:
            df = (fx_curr - fx_prev) / (x_curr - x_prev)
        else:
            # Fallback to finite difference
            h_fd = max(1e-8, abs(x_curr) * 1e-6)
            try:
                fx_plus = call_f(x_curr + h_fd)
                if math.isfinite(fx_plus):
                    df = (fx_plus - fx_curr) / h_fd
                else:
                    break
            except:
                break
        
        if not math.isfinite(df) or abs(df) < 1e-300:
            # Derivative is bad, try bisection-like exploration?
            # Or just break and return best
            break
            
        # 2. Newton Step
        step = fx_curr / df
        x_new = x_curr - step
        
        # 3. Trust Region Constraint
        # If step is outside [x_curr - delta, x_curr + delta], clamp
        # Actually, standard TR: step s such that |s| <= delta
        # Here step is the displacement.
        if abs(step) > delta:
            # Clamp to boundary
            if step > 0:
                x_new = x_curr + delta
            else:
                x_new = x_curr - delta
            
            # If clamped, the predicted reduction is different.
            # For linear model m(s) = fx + df*s
            # Predicted reduction: fx - m(s) = -df*s
            # If we clamp, s = sign(step)*delta
            # We need to be careful with acceptance criteria.
            # Simplified: If clamped, we accept if |f_new| < |f_curr|
            # Otherwise, we shrink delta.
            
            # Let's evaluate first
            try:
                fx_new = call_f(x_new)
            except:
                break
                
            if not math.isfinite(fx_new):
                delta *= 0.5
                continue
                
            # Update best
            if abs(fx_new) < best_abs_fx:
                best_x = x_new
                best_fx = fx_new
                best_abs_fx = abs(fx_new)
                
            if abs(fx_new) <= TOL:
                return x_new
                
            # Acceptance for clamped step:
            # If it improved, accept and maybe expand delta?
            # If it didn't improve, shrink delta.
            if abs(fx_new) < abs(fx_curr):
                # Accept
                x_prev, fx_prev = x_curr, fx_curr
                x_curr, fx_curr = x_new, fx_new
                # Expand delta cautiously
                delta = min(delta * 1.5, max(1e-2, abs(x_curr)*10))
            else:
                # Reject
                delta *= 0.5
                # Keep x_curr, update x_prev to x_new? 
                # No, usually we keep the last accepted point.
                # But x_new is a new evaluation. We can use it for slope next time?
                # If we reject, we stay at x_curr. 
                # We can set x_prev to x_new to get a better slope towards the "wrong" side?
                # Or just keep old slope.
                # Let's update x_prev to x_new to avoid repeating the same slope if it's bad.
                x_prev, fx_prev = x_new, fx_new
                
            continue # Loop to next iteration
            
        # 4. Unclamped Step (Standard Newton/Secant)
        # Predicted reduction: fx_curr (since m(0)=fx, m(step)=0 ideally)
        # Actual reduction: fx_curr - fx_new
        
        try:
            fx_new = call_f(x_new)
        except:
            break
            
        if not math.isfinite(fx_new):
            delta *= 0.5
            continue
            
        # Update best
        if abs(fx_new) < best_abs_fx:
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs(fx_new)
            
        if abs(fx_new) <= TOL:
            return x_new
            
        # Acceptance Criteria
        # rho = (fx_curr - fx_new) / (fx_curr - 0) = 1 - fx_new/fx_curr
        # If fx_curr and fx_new have same sign, rho < 1.
        # If fx_new is closer to 0, rho > 0.
        
        # Simple heuristic for 1D TR:
        # Accept if |fx_new| < |fx_curr|
        # If accepted, expand delta.
        # If rejected, shrink delta.
        
        if abs(fx_new) < abs(fx_curr):
            # Accept
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new
            # Expand delta
            # If step was small relative to delta, expand more
            if abs(step) < delta * 0.5:
                delta = min(delta * 2.0, max(1e-1, abs(x_curr)*100))
            else:
                delta = min(delta * 1.25, max(1e-1, abs(x_curr)*100))
        else:
            # Reject
            delta *= 0.5
            # Update previous point to the rejected one to help slope estimation?
            # Yes, the slope between x_curr and x_new is more relevant than x_curr and old x_prev
            x_prev, fx_prev = x_new, fx_new
            
        # Convergence check on delta
        if delta < 1e-15:
            break

    return best_x

# EVOLVE-BLOCK-END
