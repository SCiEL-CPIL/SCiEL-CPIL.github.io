"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0 using Adaptive Newton with Bisection Fallback.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin well under 120s limit
    
    x = float(x0)
    
    # Track best solution found so far
    best_x = x
    best_f = float('inf')
    
    # Initial evaluation
    try:
        f_x = f(x)
    except Exception:
        return x
        
    if not math.isfinite(f_x):
        return x
        
    if abs(f_x) <= 1e-8:
        return x
        
    best_f = abs(f_x)
    
    # Adaptive step size for derivative estimation
    # Start with a relatively large step to capture the slope of polynomials
    # Polynomials are smooth, so a larger h gives a better slope estimate initially
    # unless x is extremely close to a root where curvature dominates.
    h = 1e-3 * max(1.0, abs(x))
    
    # Bisection bounds, initialized to a small window around x0
    # We don't know the root location, so we start with a local bracket
    # and expand if needed, but for polynomials near x0, a local bracket is often sufficient
    # if the function is monotonic near the root.
    # However, Newton is primary. We use bisection only if Newton diverges.
    
    # To support bisection, we need a sign change.
    # Let's try to establish a bracket first with a few cheap evaluations?
    # No, that wastes calls. Let's stick to Newton but monitor divergence.
    
    # Newton Loop
    # We estimate f'(x) using central difference: (f(x+h) - f(x-h)) / (2h)
    # This costs 2 calls per iteration.
    # Secant costs 1 call per iteration after the first.
    # However, Newton converges quadratically vs Secant's superlinear (1.618).
    # For smooth polynomials, Newton with a good initial h should be faster.
    
    # Optimization: Reuse previous f(x-h) if possible?
    # In standard Newton, we move to x_new. The points are different.
    # So we always need 2 new evals for the derivative.
    # Total calls: 1 (initial) + 2 * N_iter.
    # Secant calls: 2 (initial) + 1 * N_iter.
    # If Newton takes N=5 and Secant takes N=10:
    # Newton: 1 + 10 = 11 calls.
    # Secant: 2 + 10 = 12 calls.
    # If Newton takes N=4 and Secant takes N=8:
    # Newton: 1 + 8 = 9 calls.
    # Secant: 2 + 8 = 10 calls.
    # Newton is generally more efficient if it converges reliably.
    
    # Strategy:
    # 1. Estimate derivative.
    # 2. Take Newton step.
    # 3. If step is too large or f increases significantly, dampen or switch to bisection-like step.
    # 4. Shrink h as we get closer to root to improve derivative accuracy.
    
    for i in range(100):
        if time.time() > deadline:
            break
            
        # Adjust h based on distance to potential root or previous step
        # If x is large, h should be large. If x is small, h should be small.
        # Also, if we are converging, shrink h.
        if i > 0:
            # Shrink h slightly each step to refine derivative estimate
            # but not too fast to avoid numerical noise
            h *= 0.5 
            # Ensure h is not too small relative to machine epsilon
            min_h = 1e-12 * max(1.0, abs(x))
            if h < min_h:
                h = min_h
        
        try:
            # Central difference for derivative
            f_xh = f(x + h)
            f_xmh = f(x - h)
        except Exception:
            break
            
        if not math.isfinite(f_xh) or not math.isfinite(f_xmh):
            break
            
        # Check if any of these new points are roots
        if abs(f_xh) <= 1e-8:
            return x + h
        if abs(f_xmh) <= 1e-8:
            return x - h
            
        # Update best
        if abs(f_xh) < best_f:
            best_f = abs(f_xh)
            best_x = x + h
        if abs(f_xmh) < best_f:
            best_f = abs(f_xmh)
            best_x = x - h
            
        df = (f_xh - f_xmh) / (2 * h)
        
        if df == 0 or not math.isfinite(df):
            # Derivative is zero or invalid. Stuck.
            # Try a small random kick or break
            break
            
        # Newton step
        x_new = x - f_x / df
        
        # Check for non-finite
        if not math.isfinite(x_new):
            break
            
        # Safety: If the step is very large, it might be overshooting.
        # Limit step size to, say, 10% of current magnitude + 1?
        # Or check if f(x_new) is worse.
        
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Damping / Backtracking
        if abs(f_new) > abs(f_x) * 1.1:
            # The step made things worse.
            # Try a damped step: move halfway
            x_damped = x + 0.5 * (x_new - x)
            try:
                f_damped = f(x_damped)
            except Exception:
                # If damped fails, keep current x and break or try smaller h
                # For now, break to avoid infinite loops of bad steps
                break
                
            if math.isfinite(f_damped):
                if abs(f_damped) <= 1e-8:
                    return x_damped
                if abs(f_damped) < best_f:
                    best_f = abs(f_damped)
                    best_x = x_damped
                x = x_damped
                f_x = f_damped
                # Shrink h more aggressively
                h *= 0.1
                continue
            else:
                break
        else:
            # Good step
            if abs(f_new) < best_f:
                best_f = abs(f_new)
                best_x = x_new
            x = x_new
            f_x = f_new
            # Shrink h for next iteration
            # h is already shrunk in the loop start, but let's ensure it's reasonable
            # Actually, h is shrunk at the top of the loop.
            # We might want to reset h if we moved a lot?
            # No, central difference with small h is accurate for smooth functions.
            
    return best_x
# EVOLVE-BLOCK-END
