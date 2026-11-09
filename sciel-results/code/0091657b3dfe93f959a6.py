import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= 1e-8:
        return x0

    # Phase 1: Find a bracket [a, b] such that f(a) and f(b) have opposite signs
    a = x0
    fa = f0
    b = None
    fb = None
    
    step = 1.0
    # Try to find a bracket by expanding step size
    for _ in range(50):
        # Try positive direction
        b_try = a + step
        fb_try = f(b_try)
        if math.isfinite(fb_try) and fa * fb_try < 0:
            b = b_try
            fb = fb_try
            break
            
        # Try negative direction
        b_try = a - step
        fb_try = f(b_try)
        if math.isfinite(fb_try) and fa * fb_try < 0:
            b = b_try
            fb = fb_try
            break
            
        step *= 2.0

    # If no bracket found, return best guess (likely failure, but avoids exception)
    if b is None:
        return x0

    # Phase 2: Illinois Algorithm
    # Illinois is a variant of Bisection that uses linear interpolation
    # to determine the next point, but halves the function value of the
    # endpoint with the smaller magnitude to prevent slow convergence
    # when the function is nearly linear.
    
    for _ in range(100):
        # Check if we've converged
        if abs(fb - fa) < 1e-15:
            # Function values are too close, use midpoint
            c = (a + b) / 2.0
        else:
            # Linear interpolation (Secant step)
            c = a - fa * (b - a) / (fb - fa)
            
        # If c is outside [a, b], use midpoint (Bisection step)
        if not (a < c < b):
            c = (a + b) / 2.0
            
        fc = f(c)
        
        if not math.isfinite(fc):
            # Fallback to midpoint if evaluation fails
            c = (a + b) / 2.0
            fc = f(c)
            if not math.isfinite(fc):
                break

        if abs(fc) <= 1e-8:
            return c

        # Update bracket
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc

        # Illinois modification:
        # Halve the function value of the endpoint with the smaller magnitude
        # This forces the next interpolation to be closer to the midpoint
        # of the interval, preventing the method from getting stuck in a
        # corner if the function is nearly linear.
        if abs(fa) < abs(fb):
            fa = fa / 2.0
        else:
            fb = fb / 2.0

        # Check for convergence in x
        if abs(b - a) < 1e-15 * max(1.0, abs(b)):
            break

    # Return the endpoint with the smaller function magnitude
    if abs(fa) < abs(fb):
        return a
    else:
        return b
# EVOLVE-BLOCK-END
