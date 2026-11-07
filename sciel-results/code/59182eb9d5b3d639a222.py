import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    f0 = f(x0)
    if not math.isfinite(f0):
        return x0
    if abs(f0) <= 1e-8:
        return x0

    # Initialize a second point
    # Choose a step size relative to x0
    # If x0 is 0, step 1. If x0 is 1000, step 1 is small, 1000 is large.
    # Let's try 1.0 first.
    a = x0
    fa = f0
    
    b = x0 + 1.0
    fb = f(b)
    
    if not math.isfinite(fb):
        # Try a smaller step if the function blew up
        b = x0 + 1e-6
        fb = f(b)
        
    if not math.isfinite(fb):
        # Try a larger step? Or just return x0 if it's the best we have
        # If x0 is finite and f0 is finite, x0 is a valid return if nothing else works
        pass

    if math.isfinite(fb) and abs(fb) <= 1e-8:
        return b
        
    bracketed = False
    if math.isfinite(fb) and fa * fb < 0:
        bracketed = True
    else:
        # Expand bracket
        step = 1.0
        # If the first step failed to bracket, try expanding
        # We already tried step=1.0 (b = x0+1). 
        # If that didn't bracket, try 2, 4, 8...
        for _ in range(50):
            if time.time() > deadline:
                break
            step *= 2.0
            b = x0 + step
            fb = f(b)
            if not math.isfinite(fb):
                continue
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                bracketed = True
                break
            if abs(b) > 1e12:
                break
        
        # If still not bracketed, we proceed with Secant on a, b
        # But if b is not finite, we are in trouble.
        if not math.isfinite(fb):
             # Fallback: just return x0? Or try random?
             # Task implies roots exist. If we can't find a sign change, 
             # maybe x0 is on the other side of a pole? But it's a polynomial.
             # Polynomials don't have poles. If no sign change, no root in [a,b].
             # We assume the root is reachable.
             pass

    # Ensure we have two finite points for Secant
    if not math.isfinite(fb):
        # Try to find any other finite point
        for i in range(1, 10):
            b_try = x0 + i * 0.1
            fb_try = f(b_try)
            if math.isfinite(fb_try):
                b, fb = b_try, fb_try
                break
        if not math.isfinite(fb):
            return x0

    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        # Secant step
        if fa == fb:
            if bracketed:
                x_new = (a + b) / 2.0
            else:
                x_new = b + 1e-6
        else:
            x_new = b - fb * (b - a) / (fb - fa)
            
        # If bracketed, ensure x_new is within [a, b]
        if bracketed:
            lo = min(a, b)
            hi = max(a, b)
            if x_new <= lo or x_new >= hi:
                x_new = (lo + hi) / 2.0
                
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # If we diverged
            if bracketed:
                # Bisect
                x_new = (a + b) / 2.0
                f_new = f(x_new)
                if not math.isfinite(f_new):
                    break
            else:
                # Try to shrink step? Or just break
                break

        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Update points
        if bracketed:
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
        else:
            # Move window forward
            a, fa = b, fb
            b, fb = x_new, f_new
            # Check if we accidentally found a bracket
            if fa * fb < 0:
                bracketed = True

    return best_x
