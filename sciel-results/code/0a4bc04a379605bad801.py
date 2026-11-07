import time
import math

_DEADLINE = 140.0  # safety margin before 150s kill

def find_root(f, x0):
    t0 = time.monotonic()
    x = float(x0)
    
    # Evaluate initial f(x)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Store for reuse
    prev_x = x
    prev_fx = fx
    
    # Phase 1: Newton's method with forward-difference derivative
    # Reuse f(x) from previous iteration to minimize calls
    converged = False
    for i in range(50):
        if time.monotonic() - t0 > _DEADLINE:
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Forward diff to save calls: 1 extra call
        h = 1e-7 * max(1.0, abs(x))
        fxh = f(x + h)
        d = (fxh - fx) / h
        
        if abs(d) < 1e-30:
            # Derivative near zero, try secant step
            if abs(prev_x - x) > 1e-15:
                d_sec = (fx - prev_fx) / (x - prev_x)
                if abs(d_sec) > 1e-30:
                    x_new = x - fx / d_sec
                else:
                    # Nudge
                    x_new = x + 1e-6
            else:
                x_new = x + 1e-6
        else:
            x_new = x - fx / d
            # Damping: if step is too large, reduce it
            if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
                x_new = x + 10.0 * max(1.0, abs(x)) * (1 if x_new > x else -1)
        
        # Check for divergence or stall
        if abs(fxh) > 10 * abs(fx) or not math.isfinite(x_new):
            # Newton is diverging, break out to bracket
            break
            
        prev_x = x
        prev_fx = fx
        x = x_new
        fx = f(x)
        
        if abs(fx) <= 1e-8:
            converged = True
            break

    if converged:
        return x

    # Phase 2: Bracketing if Newton failed
    # Try to find an interval [a, b] such that f(a)*f(b) < 0
    a = x
    fa = fx
    b = x0
    fb = prev_fx if abs(prev_x - x0) > 1e-15 else f(x0)
    
    # If we don't have a bracket yet, expand
    if fa * fb > 0:
        # Expand interval around x
        step = max(1.0, abs(x)) * 0.1
        for _ in range(20):
            if time.monotonic() - t0 > _DEADLINE:
                break
            # Try expanding to the right
            b_new = x + step
            fb_new = f(b_new)
            if fa * fb_new < 0:
                a, b = x, b_new
                fa, fb = fx, fb_new
                break
            # Try expanding to the left
            a_new = x - step
            fa_new = f(a_new)
            if fa_new * fb < 0:
                a, b = a_new, x
                fa, fb = fa_new, fx
                break
            step *= 2.0
    
    # Phase 3: Bisection/Secant Hybrid
    # Use the bracket [a, b]
    if fa * fb < 0:
        for _ in range(100):
            if time.monotonic() - t0 > _DEADLINE:
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
                
            # Secant step
            if abs(fb - fa) > 1e-15:
                x_sec = b - fb * (b - a) / (fb - fa)
            else:
                x_sec = (a + b) / 2.0
            
            # Bisection step
            x_bis = (a + b) / 2.0
            
            # Choose the point closer to the midpoint (standard hybrid)
            # Or just use bisection if secant is outside bracket
            if a < x_sec < b:
                x_mid = x_sec
            else:
                x_mid = x_bis
                
            f_mid = f(x_mid)
            
            if abs(f_mid) <= 1e-8:
                return x_mid
                
            if fa * f_mid < 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
                
        # Return best of a, b, x_mid
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    # Fallback: Return the best point found so far
    if abs(fx) <= abs(fa) and abs(fx) <= abs(fb):
        return x
    if abs(fa) < abs(fb):
        return a
    return b
