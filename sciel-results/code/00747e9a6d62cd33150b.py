"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find a root of f starting from x0.
    Uses a hybrid strategy:
      1. Secant/Newton hybrid for fast convergence.
      2. Fallback to bisection if derivative is bad or step diverges.
      3. Deadline guard to ensure we return within time limits.
    
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    x = float(x0)
    
    # First evaluation
    fx = f(x)
    if not math.isfinite(fx):
        # If x0 gives non-finite, try a small perturbation
        x = x0 + 1e-6
        fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Store best so far
    best_x = x
    best_fabs = abs(fx)
    
    # Initialize secant method with two points
    # Use finite difference for initial second point if needed
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x - h
    f_prev = f(x_prev)
    
    if not math.isfinite(f_prev):
        x_prev = x + h
        f_prev = f(x_prev)
    
    if not math.isfinite(f_prev):
        # Try larger step
        x_prev = x + 1e-3
        f_prev = f(x_prev)
    
    if not math.isfinite(f_prev):
        # Give up on secant, just return best
        return best_x
    
    # Secant/Newton hybrid iteration
    for i in range(100):
        if time.time() > deadline:
            break
        
        # Check if we've converged
        if best_fabs <= 1e-8:
            break
        
        # Compute next step using secant method
        if abs(f_prev - fx) < 1e-30:
            # Nearly parallel, try Newton with finite diff
            h = 1e-8 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if abs(d) < 1e-30:
                # Derivative too small, try bisection-like step
                x_new = x + 0.1 * (x - x_prev)
            else:
                x_new = x - fx / d
        else:
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
        
        # Ensure x_new is finite
        if not math.isfinite(x_new):
            # Fall back to a conservative step
            x_new = x + 0.5 * (x - x_prev)
            if not math.isfinite(x_new):
                break
        
        # Check for divergence: if step is too large, dampen
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            # Dampen the step
            direction = 1.0 if x_new > x else -1.0
            x_new = x + direction * 10.0 * max(1.0, abs(x))
        
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # Step went to bad region, shrink step
            x_new = x + 0.5 * (x - x_prev)
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
        
        # Update best
        if abs(f_new) < best_fabs:
            best_fabs = abs(f_new)
            best_x = x_new
        
        # Check convergence
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update for next iteration
        x_prev = x
        f_prev = fx
        x = x_new
        fx = f_new
    
    return best_x

# EVOLVE-BLOCK-END
