"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8 using a safeguarded inverse-interpolation
    method (Brent-style) that reuses the two most recent evaluations per step.
    """
    deadline = time.time() + 110.0  # safety margin before 120s wall-clock
    x = float(x0)
    
    # First evaluation
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    if not math.isfinite(fx):
        # Try a small perturbation
        x = x + 1e-4 * max(1.0, abs(x))
        fx = f(x)
        if abs(fx) <= 1e-8:
            return x
        if not math.isfinite(fx):
            return x  # give up; return best guess
    
    # We will maintain a bracket [a, b] if possible, but start with just x.
    # Use a hybrid approach: start with Newton-like steps but safeguard with bisection.
    # Actually, let's use a proper safeguarded method.
    # Since we don't have a bracket initially, we'll try to find one quickly.
    
    # Try to find a bracket around x0
    a = x
    fa = fx
    step = 1.0
    b = None
    fb = None
    
    # Expand to find a sign change
    for _ in range(50):
        if time.time() > deadline:
            return x if abs(fa) <= abs(fx) else x
        b = a + step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # If no sign change, try the other direction
        b = a - step
        fb = f(b)
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
    else:
        # No bracket found; fall back to Newton with finite differences
        for _ in range(30):
            if time.time() > deadline:
                break
            h = 1e-7 * max(1.0, abs(a))
            d = (f(a + h) - fa) / h
            if d == 0 or not math.isfinite(d):
                break
            new_a = a - fa / d
            new_fa = f(new_a)
            if abs(new_fa) < abs(fa):
                a, fa = new_a, new_fa
                if abs(fa) <= 1e-8:
                    return a
            else:
                break
        return a
    
    # Now we have a bracket [a, b] with fa * fb <= 0
    # Ensure a is the left endpoint
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Safeguarded inverse interpolation (Brent's method simplified)
    # We keep track of the two most recent points for interpolation
    c = a
    fc = fa
    d = b
    fd = fb
    
    # The root is in [c, d] where fc * fd <= 0
    # Use inverse quadratic interpolation if possible, else bisection
    
    best_x = c if abs(fc) < abs(fd) else d
    best_f = min(abs(fc), abs(fd))
    
    for _ in range(100):
        if time.time() > deadline:
            break
        if abs(fd) <= 1e-8:
            return d
        if abs(fc) <= 1e-8:
            return c
        
        # Try inverse quadratic interpolation
        # Points: (c, fc), (d, fd)
        # We need a third point for quadratic; use the previous value of c or d
        # For simplicity, use inverse linear interpolation (Regula Falsi) with safeguard
        # x_new = (c*fd - d*fc) / (fd - fc)
        
        denom = fd - fc
        if abs(denom) < 1e-300:
            # Bisection
            x_new = (c + d) / 2.0
        else:
            x_new = (c * fd - d * fc) / denom
        
        # Safeguard: x_new must be in (c, d)
        if not (c < x_new < d):
            x_new = (c + d) / 2.0
        
        f_new = f(x_new)
        if not math.isfinite(f_new):
            # Bisection
            x_new = (c + d) / 2.0
            f_new = f(x_new)
        
        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update bracket
        if fc * f_new <= 0:
            d = x_new
            fd = f_new
        else:
            c = x_new
            fc = f_new
        
        # Check convergence
        if abs(d - c) < 1e-14 * max(1.0, abs(c), abs(d)):
            break
    
    # Return the best point found
    if abs(fd) <= abs(fc):
        return d
    else:
        return c
# EVOLVE-BLOCK-END
