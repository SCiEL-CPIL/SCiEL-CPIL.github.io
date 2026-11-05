"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def find_root(f, x0):
    """
    Robust root finder using a combination of techniques:
    1. Try Newton's method with adaptive step control
    2. Fall back to bisection if Newton fails or diverges
    3. Use secant method as intermediate
    4. Multiple starting points if initial guess is poor
    """
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    # Try Newton's method with damping
    x_newton = x
    fx_newton = fx
    for i in range(30):
        if abs(fx_newton) <= 1e-8:
            return x_newton
        
        # Compute derivative using finite differences
        h = 1e-7 * max(1.0, abs(x_newton))
        # Use central difference for better accuracy
        fp = f(x_newton + h)
        fm = f(x_newton - h)
        d = (fp - fm) / (2 * h)
        
        if d == 0 or not np.isfinite(d):
            break
        
        # Damped Newton step
        step = fx_newton / d
        # Limit step size to prevent divergence
        max_step = 10 * max(1.0, abs(x_newton))
        if abs(step) > max_step:
            step = max_step * np.sign(step)
        
        x_candidate = x_newton - step
        fx_candidate = f(x_candidate)
        
        # Accept step if it reduces |f|
        if abs(fx_candidate) < abs(fx_newton) * 1.1:
            x_newton = x_candidate
            fx_newton = fx_candidate
        else:
            # Try half step
            x_candidate = x_newton - step / 2
            fx_candidate = f(x_candidate)
            if abs(fx_candidate) < abs(fx_newton) * 1.1:
                x_newton = x_candidate
                fx_newton = fx_candidate
            else:
                break
    
    if abs(fx_newton) <= 1e-8:
        return x_newton
    
    # If Newton failed, try secant method from two points
    x1 = x_newton
    f1 = fx_newton
    h_sec = 1e-4 * max(1.0, abs(x1))
    x2 = x1 + h_sec
    f2 = f(x2)
    
    for i in range(40):
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
        
        if f2 == f1:
            break
        
        # Secant step
        x_sec = x2 - f2 * (x2 - x1) / (f2 - f1)
        f_sec = f(x_sec)
        
        if not np.isfinite(f_sec):
            break
        
        x1, f1 = x2, f2
        x2, f2 = x_sec, f_sec
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2
    
    # Fall back to bisection: find a bracket around the root
    # First, try to find a sign change
    a = min(x1, x2)
    b = max(x1, x2)
    fa = f(a)
    fb = f(b)
    
    # Expand bracket if no sign change
    width = b - a
    if width == 0:
        width = 1.0
    expansion = width
    for _ in range(50):
        if fa * fb < 0:
            break
        a_new = a - expansion
        b_new = b + expansion
        fa_new = f(a_new)
        fb_new = f(b_new)
        if fa_new * fa < 0:
            b, fb = a, fa
            a, fa = a_new, fa_new
            break
        elif fb * fb_new < 0:
            a, fa = b, fb
            b, fb = b_new, fb_new
            break
        else:
            a, fa = a_new, fa_new
            b, fb = b_new, fb_new
            expansion *= 2
    
    if fa * fb < 0:
        # Bisection
        for _ in range(60):
            mid = (a + b) / 2
            fmid = f(mid)
            if abs(fmid) <= 1e-8:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
            if b - a < 1e-12:
                break
        return (a + b) / 2
    
    # Last resort: return the best estimate
    candidates = [x, x_newton, x1, x2]
    best_x = min(candidates, key=lambda z: abs(f(z)))
    return best_x

# EVOLVE-BLOCK-END
