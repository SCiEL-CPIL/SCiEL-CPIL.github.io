"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # 2. Safeguarded Newton-Raphson Method
    # We want to minimize |f(x)|. 
    # Newton step: x_new = x - f(x)/f'(x)
    # We approximate f'(x) using central difference if possible, or forward.
    # However, for polynomials, we can just use a standard Newton step with a safeguard.
    # If the Newton step is too large or moves away, we take a smaller step.
    
    # Initial step size for safeguarding
    # We will track the best x and fx found so far.
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    # Adaptive step size for the "safe" fallback
    # We'll use a simple strategy: 
    # Try Newton. If it fails (non-finite, or |f| increases significantly), 
    # try a bisection-like step towards a point where sign might change, 
    # or just a small step in the direction of decreasing |f|? 
    # Actually, for smooth polynomials, if we don't have a bracket, 
    # a robust approach is to use the Secant method with a fallback, 
    # or simply a very robust Newton with line search on |f|.
    
    # Let's implement a Newton method with a line search on g(x) = f(x)^2.
    # This is equivalent to minimizing |f(x)|.
    
    h = 1e-6  # Step for derivative estimation
    x_prev = None
    f_prev = None
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Check if we have a second point for Secant (more robust than FD derivative for some cases)
        # But Newton with FD derivative is standard. Let's stick to Newton with FD derivative.
        
        # Estimate derivative
        # Central difference is more accurate but costs 2 calls. Forward costs 1.
        # Since we care about call count, let's use forward difference if possible, 
        # or reuse previous evaluations? 
        # Actually, for smooth polynomials, a simple 2-point forward difference is often enough.
        # x_h = x + h
        # f_h = f(x_h)
        # f_prime = (f_h - fx) / h
        
        # To save calls, we can try to estimate derivative with a smaller h if we are close.
        # But let's just use a fixed small h for derivative estimation.
        
        x_h = x + 1e-5
        try:
            f_h = f(x_h)
        except Exception:
            break
            
        if not math.isfinite(f_h):
            # Try smaller h
            x_h = x + 1e-8
            try:
                f_h = f(x_h)
            except Exception:
                break
            if not math.isfinite(f_h):
                # Try backward
                x_h = x - 1e-5
                try:
                    f_h = f(x_h)
                except Exception:
                    break
                if not math.isfinite(f_h):
                    break

        f_prime = (f_h - fx) / 1e-5
        
        if not math.isfinite(f_prime) or abs(f_prime) < 1e-12:
            # Derivative is zero or non-finite. 
            # If derivative is zero, we are at a local extremum of f. 
            # If |f| is small, we might be close to a root where f touches 0.
            # Otherwise, we need to move away.
            # Let's take a small step in a random direction or towards x0?
            # Better: try a secant step if we have a previous point.
            if x_prev is not None and f_prev is not None and abs(f_prev - fx) > 1e-12:
                # Secant method
                x_new = x - fx * (x - x_prev) / (fx - f_prev)
            else:
                # Just take a small step
                x_new = x + 1e-3
        else:
            # Newton step
            x_new = x - fx / f_prime
            
        # Safeguard: Ensure x_new is not too far from x
        # If the step is too large, shrink it.
        # Heuristic: limit step to 10 * current scale or something.
        # Since we don't know the scale, let's limit the step relative to x.
        # If x is 0, limit to 1.0. If x is 100, limit to 10.
        max_step = max(1.0, abs(x) * 0.5)
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
            
        try:
            fx_new = f(x_new)
        except Exception:
            # If evaluation fails, shrink step
            x_new = x + 0.5 * (x - x_prev) if x_prev is not None else x + 1e-4
            try:
                fx_new = f(x_new)
            except Exception:
                break
        
        if not math.isfinite(fx_new):
            # Shrink step
            x_new = x + 0.5 * (x - x_prev) if x_prev is not None else x + 1e-4
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break

        # Accept if |f| decreases
        if abs(fx_new) < best_abs_fx:
            best_x = x_new
            best_fx = fx_new
            best_abs_fx = abs(fx_new)
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Update for next iteration
        # We only move if the new point is better than the old one in terms of |f|?
        # Newton method usually moves even if |f| increases, as long as it converges.
        # But for robustness, if |f| increases significantly, we might want to backtrack.
        
        # Simple rule: Always move to x_new if it is finite.
        # But if |f_new| > |f_old| * 1.1, it might be diverging.
        # Let's use a simple line search idea: if |f_new| > |f_old|, shrink step.
        
        if abs(fx_new) > abs(fx):
            # Backtrack
            x_new = x + 0.5 * (x_new - x)
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(fx_new):
                break
            # If still bad, give up on this direction?
            # For now, just accept it and let the next iteration handle it, 
            # or break if it's significantly worse.
            if abs(fx_new) > abs(fx) * 1.5:
                # This direction is bad. Try the other way?
                # For simplicity, just break and rely on the best_x found so far.
                # Or try a secant step.
                pass

        x_prev = x
        f_prev = fx
        x = x_new
        fx = fx_new
        
        if abs(fx) <= 1e-8:
            return x

    # 3. If Newton didn't converge, return the best point found
    if abs(best_fx) <= 1e-8:
        return best_x
        
    # 4. Final fallback: Try a few random nearby points? 
    # Or just return best_x.
    return best_x
# EVOLVE-BLOCK-END
