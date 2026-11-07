import math

def find_root(f, x0):
    """
    Finds a root of f starting from x0.
    Returns x such that |f(x)| <= 1e-8.
    """
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = f(x)
    except Exception:
        return x
        
    if not math.isfinite(fx):
        return x
        
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Steffensen's Method (derivative-free, quadratic convergence)
    # Requires 2 function evaluations per iteration (x and x+f(x))
    x_curr = x
    fx_curr = fx
    for _ in range(30):
        if abs(fx_curr) <= 1e-8:
            return x_curr
            
        # Compute the next point
        x_plus = x_curr + fx_curr
        try:
            fx_plus = f(x_plus)
        except Exception:
            break
            
        if not math.isfinite(fx_plus):
            break
            
        # Check for division by zero or near-zero
        denom = fx_plus - fx_curr
        if abs(denom) < 1e-30:
            break
            
        # Steffensen update: x_new = x - fx^2 / (fx_plus - fx)
        x_new = x_curr - (fx_curr * fx_curr) / denom
        
        if not math.isfinite(x_new):
            break
            
        # Check convergence
        if abs(x_new - x_curr) <= 1e-12 * max(1.0, abs(x_new)):
            # Verify the root
            try:
                f_new = f(x_new)
                if math.isfinite(f_new) and abs(f_new) <= 1e-8:
                    return x_new
            except Exception:
                pass
            break
            
        x_curr = x_new
        fx_curr = fx_plus

    # Strategy 2: Bisection-like search with Secant fallback
    # Try to find a bracket first by expanding
    a, b = None, None
    fa = fx
    step = 1.0
    max_expansions = 20
    
    for _ in range(max_expansions):
        x_plus = x + step
        try:
            fx_plus = f(x_plus)
            if math.isfinite(fx_plus) and fa * fx_plus < 0:
                a, b = x, x_plus
                break
        except Exception:
            pass
            
        x_minus = x - step
        try:
            fx_minus = f(x_minus)
            if math.isfinite(fx_minus) and fa * fx_minus < 0:
                a, b = x_minus, x
                break
        except Exception:
            pass
            
        step *= 2.0

    if a is not None and b is not None:
        # Use Bisection for guaranteed convergence (slower but robust)
        fa = f(a)
        fb = f(b)
        for _ in range(100):
            mid = (a + b) / 2.0
            fm = f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) <= 1e-12:
                break
        # Return the best point
        return a if abs(fa) < abs(fb) else b

    # Strategy 3: Illinois Algorithm (Secant with damping)
    x1 = x
    f1 = fx
    x2 = x + 1e-6 * max(1.0, abs(x))
    try:
        f2 = f(x2)
    except Exception:
        return x
        
    if not math.isfinite(f2):
        return x

    for _ in range(100):
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
            
        if f2 == f1:
            x2 = x1 + 1e-6
            try:
                f2 = f(x2)
            except Exception:
                return x1
            continue

        x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
        
        if not math.isfinite(x_new):
            x_new = (x1 + x2) / 2.0
            
        try:
            f_new = f(x_new)
        except Exception:
            return x1
            
        if not math.isfinite(f_new):
            return x1

        # Illinois rule
        if (x_new - x1) * (x2 - x1) > 0 and (f_new - f1) * (f2 - f1) > 0:
            x2 = (x1 + x2) / 2.0
            try:
                f2 = f(x2)
            except Exception:
                return x1
            if not math.isfinite(f2):
                return x1
        else:
            x2 = x1
            f2 = f1
            x1 = x_new
            f1 = f_new

    return x1 if abs(f1) < abs(f2) else x2
