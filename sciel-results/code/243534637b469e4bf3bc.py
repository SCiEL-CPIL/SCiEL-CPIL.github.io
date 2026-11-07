import math
import time

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 using the Secant method.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    x = float(x0)
    fx = f(x)
    
    # If we are already close enough, return immediately
    if abs(fx) <= 1e-8:
        return x
        
    # Initialize second point
    # Use a small step relative to x, but not too small
    h = 1e-4 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + h
    f_curr = f(x_curr)
    
    # If the second point is better, swap or just proceed
    # Actually, let's just proceed with Secant.
    # If f_curr is 0, we found a root
    if abs(f_curr) <= 1e-8:
        return x_curr
        
    # Secant Iteration
    # We have x_prev, f_prev (which is fx) and x_curr, f_curr
    f_prev = fx
    
    for i in range(100):
        if time.time() > deadline:
            # Return the best candidate found so far
            if abs(f_prev) < abs(f_curr):
                return x_prev
            else:
                return x_curr

        # Check for convergence of f_curr
        if abs(f_curr) <= 1e-8:
            return x_curr
            
        # Calculate next point
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            # Flat region or very close points, nudge x_curr
            x_curr = x_curr + 1e-4 * max(1.0, abs(x_curr))
            f_curr = f(x_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
            # Update prev to current to restart slope calc
            x_prev, f_prev = x_curr, f_curr
            continue
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Optional: Bisection check if we have a bracket
        # For now, pure Secant is usually faster for smooth polys.
        
        f_new = f(x_new)
        
        # Update
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new
        
    # If loop finishes, return the best one
    if abs(f_prev) < abs(f_curr):
        return x_prev
    else:
        return x_curr
