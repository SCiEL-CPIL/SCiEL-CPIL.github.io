"""Root finder for 3 fixed oscillatory equations."""
import math
import time

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 for a given function f and starting point x0.
    Uses a combination of:
      1. Initial scan to find a bracket
      2. Bisection to safely narrow the bracket
      3. Newton's method (with secant fallback) for fast convergence
      4. Deadline guard to avoid timeouts
    """
    tol = 1e-8
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    
    x = float(x0)
    
    def elapsed():
        return time.time() - start_time
    
    def check_deadline():
        return elapsed() > deadline
    
    # Try the starting point first
    fx = f(x)
    if not math.isfinite(fx):
        return x
    if abs(fx) <= tol:
        return x
    
    # Phase 1: Find a bracket [a, b] with f(a)*f(b) < 0
    # Try small steps first, then larger
    a = x
    fa = fx
    
    # Try expanding in both directions
    best_a = a
    best_fa = fa
    found_bracket = False
    bracket = None
    
    # Try steps: very small to large
    steps = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
    
    for delta in steps:
        if check_deadline():
            break
        # Try x + delta
        x_plus = x + delta
        try:
            f_plus = f(x_plus)
        except Exception:
            f_plus = float('inf')
        if math.isfinite(f_plus):
            if abs(f_plus) <= tol:
                return x_plus
            if fa * f_plus < 0:
                bracket = (a, x_plus)
                found_bracket = True
                break
        if check_deadline():
            break
        # Try x - delta
        x_minus = x - delta
        try:
            f_minus = f(x_minus)
        except Exception:
            f_minus = float('inf')
        if math.isfinite(f_minus):
            if abs(f_minus) <= tol:
                return x_minus
            if fa * f_minus < 0:
                bracket = (x_minus, a)
                found_bracket = True
                break
    
    # If we found a bracket, use bisection + Newton
    if found_bracket and bracket is not None:
        lo, hi = bracket
        flo = f(lo)
        fhi = f(hi)
        
        # Ensure flo * fhi < 0
        if flo * fhi > 0:
            # Shouldn't happen, but just in case
            lo, hi = hi, lo
            flo, fhi = fhi, flo
        
        # Bisection to narrow down, then Newton for polish
        x_lo = lo
        x_hi = hi
        f_lo = flo
        f_hi = fhi
        
        for i in range(60):
            if check_deadline():
                break
            mid = 0.5 * (x_lo + x_hi)
            if not math.isfinite(mid):
                break
            f_mid = f(mid)
            if not math.isfinite(f_mid):
                # Fall back to pure bisection
                x_lo = x_hi if f_lo * f_mid > 0 else x_lo
                continue
            if abs(f_mid) <= tol:
                return mid
            if f_lo * f_mid <= 0:
                x_hi = mid
                f_hi = f_mid
            else:
                x_lo = mid
                f_lo = f_mid
            
            # Try Newton step from mid
            h = 1e-7 * max(1.0, abs(mid))
            try:
                f_mid_h = f(mid + h)
            except Exception:
                f_mid_h = float('inf')
            if math.isfinite(f_mid_h):
                deriv = (f_mid_h - f_mid) / h
                if abs(deriv) > 1e-15:
                    x_new = mid - f_mid / deriv
                    if x_lo < x_new < x_hi and math.isfinite(x_new):
                        f_new = f(x_new)
                        if math.isfinite(f_new) and abs(f_new) <= tol:
                            return x_new
                        if f_lo * f_new <= 0:
                            x_hi = x_new
                            f_hi = f_new
                        else:
                            x_lo = x_new
                            f_lo = f_new
        
        # Return the midpoint of the final bracket
        result = 0.5 * (x_lo + x_hi)
        if math.isfinite(result):
            return result
        return mid
    
    # Phase 2: No bracket found. Use Steffensen's method or secant method.
    # Try secant method with two nearby points
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    try:
        f1 = f(x1)
    except Exception:
        f1 = float('inf')
    
    if not math.isfinite(f1):
        # Try different h
        h = 1e-2 * max(1.0, abs(x))
        x1 = x + h
        try:
            f1 = f(x1)
        except Exception:
            f1 = float('inf')
    
    if not math.isfinite(f1):
        # Just return x as best guess
        return x
    
    # Secant method
    x0_sec = x
    x1_sec = x1
    f0 = fx
    f1 = f1
    
    best_x = x0_sec
    best_f = abs(f0)
    
    for i in range(80):
        if check_deadline():
            break
        if abs(f1 - f0) < 1e-15:
            break
        x2 = x1_sec - f1 * (x1_sec - x0_sec) / (f1 - f0)
        if not math.isfinite(x2):
            break
        try:
            f2 = f(x2)
        except Exception:
            f2 = float('inf')
        if not math.isfinite(f2):
            break
        if abs(f2) < best_f:
            best_x = x2
            best_f = abs(f2)
        if abs(f2) <= tol:
            return x2
        x0_sec = x1_sec
        f0 = f1
        x1_sec = x2
        f1 = f2
    
    return best_x if math.isfinite(best_x) else x
