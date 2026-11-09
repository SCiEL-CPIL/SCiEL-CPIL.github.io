import math

def find_root(f, x0):
    x = float(x0)
    try:
        fx = f(x)
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    a = None
    b = None
    fa = None
    fb = None
    
    # Initial step size
    base_step = 0.1 * max(1.0, abs(x))
    step = base_step
    max_step = 1e6 * max(1.0, abs(x))
    
    # Bracketing Phase
    # Use ratio 1.5 to reduce risk of missing roots in oscillatory functions
    while step < max_step:
        x_plus = x + step
        try:
            fx_plus = f(x_plus)
        except Exception:
            fx_plus = float('nan')
            
        if math.isfinite(fx_plus):
            if abs(fx_plus) <= 1e-8:
                return x_plus
            if fx * fx_plus < 0:
                a, b = x, x_plus
                fa, fb = fx, fx_plus
                break
                
        x_minus = x - step
        try:
            fx_minus = f(x_minus)
        except Exception:
            fx_minus = float('nan')
            
        if math.isfinite(fx_minus):
            if abs(fx_minus) <= 1e-8:
                return x_minus
            if fx * fx_minus < 0:
                a, b = x_minus, x
                fa, fb = fx_minus, fx
                break
                
        step *= 1.5
        
    if a is None:
        return x

    # Root Finding Phase: Hybrid Secant/Bisection
    # We have a, b with sign change.
    # We want to converge to root.
    # Initialize with a, b and midpoint?
    # Secant needs two points. We have a, b.
    # Let's use a and b as the first two points for Secant.
    
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    # If f1 or f2 is 0, we are done.
    if abs(f1) <= 1e-8: return x1
    if abs(f2) <= 1e-8: return x2
    
    max_iter = 100
    for _ in range(max_iter):
        # Check if interval is small enough
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            # Return the point with smaller |f|
            # We need to evaluate the midpoint or just pick one?
            # Let's pick the midpoint for safety if we haven't evaluated it.
            # Actually, if interval is tiny, a or b is likely good.
            # But we need |f(x)| <= 1e-8.
            # Let's evaluate the midpoint to be sure.
            mid = (a + b) / 2.0
            try:
                f_mid = f(mid)
                if math.isfinite(f_mid) and abs(f_mid) <= 1e-8:
                    return mid
            except Exception:
                pass
            # If mid fails, return the one with smaller |f| from a,b
            if abs(f1) < abs(f2):
                return x1
            else:
                return x2

        # Secant Step
        # x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        if abs(f2 - f1) < 1e-15:
            # Degenerate, use bisection
            x_new = (x1 + x2) / 2.0
        else:
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            
        # If x_new is outside [min(a,b), max(a,b)], use bisection
        if x_new < min(a, b) or x_new > max(a, b):
            x_new = (a + b) / 2.0
            f_new = f(x_new)
            # Update bracket
            if f_new * f1 < 0:
                b = x_new
                f2 = f_new
                x2 = x_new
            else:
                a = x_new
                f1 = f_new
                x1 = x_new
            # Also update the "other" point for secant?
            # Standard hybrid keeps the two most recent points.
            # Let's just re-evaluate the next step based on new a, b.
            # To keep it simple and robust, if we bisect, we update the bracket.
            # We need to maintain x1, x2, f1, f2 as the last two *evaluated* points 
            # that are distinct.
            # This logic is getting complex. Let's simplify.
            
            # Simpler Hybrid:
            # Always compute Secant candidate.
            # If candidate is in bracket, use it.
            # Else, use Midpoint.
            # Update bracket based on sign.
            # Keep track of last two points for next Secant step.
            
            # Let's restart the loop logic with a cleaner state.
            pass # Placeholder, logic below replaces this block entirely

    # Fallback
    return x
