"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def find_root(f, x0):
    """
    Find a root of f near x0 using Inverse Quadratic Interpolation (IQI)
    with Bisection safeguard.
    """
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # 1. Robust Bracketing
    a = x
    fa = fx
    bracket_found = False
    b = None
    fb = None
    
    # Try expanding in positive direction
    step = 1e-4
    for _ in range(50):
        b = a + step
        fb = f(b)
        if abs(fb) <= 1e-8:
            return b
        if fa * fb < 0:
            bracket_found = True
            break
        step *= 2.0
        if not math.isfinite(b):
            break
            
    if not bracket_found:
        # Try expanding in negative direction
        step = 1e-4
        for _ in range(50):
            b = a - step
            fb = f(b)
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                bracket_found = True
                break
            step *= 2.0
            if not math.isfinite(b):
                break

    if not bracket_found or not math.isfinite(b):
        # Fallback: Secant method
        x1 = a
        f1 = fa
        x2 = a + 1e-4
        f2 = f(x2)
        for _ in range(50):
            if abs(f2) <= 1e-8: return x2
            if abs(f1) <= 1e-8: return x1
            if f1 == f2: break
            x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            if not math.isfinite(x_new): break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8: return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return x1 if abs(f1) < abs(f2) else x2

    # 2. IQI Main Loop
    # Current bracket [a, b]
    # Initialize 3rd point c as midpoint
    c = (a + b) / 2.0
    fc = f(c)
    if abs(fc) <= 1e-8:
        return c

    # History: list of (x, f(x))
    # We will maintain the last 3 points.
    pts = [(a, fa), (b, fb), (c, fc)]
    
    for _ in range(100):
        # Get last 3 points
        x1, f1 = pts[0]
        x2, f2 = pts[1]
        x3, f3 = pts[2]
        
        x_new = None
        
        # IQI Formula
        d1 = (f1 - f2) * (f1 - f3)
        d2 = (f2 - f1) * (f2 - f3)
        d3 = (f3 - f1) * (f3 - f2)
        
        if abs(d1) > 1e-15 and abs(d2) > 1e-15 and abs(d3) > 1e-15:
            try:
                term1 = x1 * (f2 * f3) / d1
                term2 = x2 * (f1 * f3) / d2
                term3 = x3 * (f1 * f2) / d3
                x_new = term1 + term2 + term3
                
                # Validate x_new is within current bracket [a, b]
                if not math.isfinite(x_new) or not ((a < x_new < b) or (b < x_new < a)):
                    x_new = None
            except (ZeroDivisionError, OverflowError):
                x_new = None
        
        if x_new is None:
            # Bisection fallback
            x_new = (a + b) / 2.0
            
        fx_new = f(x_new)
        if abs(fx_new) <= 1e-8:
            return x_new
            
        # Update Bracket
        if fa * fx_new < 0:
            b = x_new
            fb = fx_new
        else:
            a = x_new
            fa = fx_new
            
        # Update History
        pts.append((x_new, fx_new))
        if len(pts) > 3:
            pts.pop(0)
            
        # Convergence check
        if abs(b - a) < 1e-12:
             return a if abs(fa) < abs(fb) else b

    # Return best estimate
    return x_new
# EVOLVE-BLOCK-END
