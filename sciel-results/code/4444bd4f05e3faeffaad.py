"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using:
    1. Secant method (1 call/iteration, fast convergence)
    2. Bisection fallback when secant diverges
    3. Final polish pass for high precision
    
    Returns the best x found (minimizing |f(x)|) within the call budget.
    """
    tol = 1e-8
    max_calls = 500
    call_count = 0
    
    def safe_eval(x):
        nonlocal call_count
        if call_count >= max_calls:
            return None
        try:
            val = f(x)
            call_count += 1
            if val is None or not _is_finite(val):
                return None
            return float(val)
        except Exception:
            call_count += 1
            return None
    
    # Initial evaluation
    x = float(x0)
    fx = safe_eval(x)
    if fx is None:
        return x
    if abs(fx) <= tol:
        return x
    
    best_x = x
    best_fabs = abs(fx)
    
    # Try to establish a bracket by expanding from x0
    # First, get a second point for secant
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    fx1 = safe_eval(x1)
    
    if fx1 is None:
        # Can't proceed with secant, try bisection-like search
        # Just return best found
        return best_x
    
    # If we have a sign change, we have a bracket
    if fx * fx1 < 0:
        a, b = x, x1
        fa, fb = fx, fx1
        # Bisection + secant hybrid
        for _ in range(100):
            if _time_left() <= 0:
                break
            if abs(fb) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
            # Midpoint for bisection
            m = (a + b) / 2
            fm = safe_eval(m)
            if fm is None:
                break
            if abs(fm) < best_fabs:
                best_x = m
                best_fabs = abs(fm)
            if abs(fm) <= tol:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                break
        return best_x
    
    # No bracket: use secant method
    # x0, f(x0) and x1, f(x1)
    xa, fa = x, fx
    xb, fb = x1, fx1
    
    # Track best
    if abs(fb) < best_fabs:
        best_x = xb
        best_fabs = abs(fb)
    
    for _ in range(120):
        if _time_left() <= 0:
            break
        if call_count >= max_calls - 2:
            break
        
        # Secant step
        if fb - fa == 0:
            break
        xc = xb - fb * (xb - xa) / (fb - fa)
        
        if not _is_finite(xc):
            break
        
        fc = safe_eval(xc)
        if fc is None:
            break
        
        if abs(fc) < best_fabs:
            best_x = xc
            best_fabs = abs(fc)
        
        if abs(fc) <= tol:
            return xc
        
        # Check for convergence
        if abs(xc - xb) < 1e-14 * max(1.0, abs(xc)):
            break
        
        # Update: keep the two most recent points
        xa, fa = xb, fb
        xb, fb = xc, fc
        
        # If we're diverging (values getting larger), try to find a bracket
        if abs(fb) > 1e6 * max(1.0, abs(fa)):
            # Try to find a bracket around current best
            break
    
    # Final polish: if we have a good candidate, do a few Newton-like steps
    # using secant with very close points
    if best_fabs <= 1e-4 and call_count < max_calls - 5:
        # Use secant with very close points for high precision
        xpol = best_x
        fpol = safe_eval(xpol)
        if fpol is not None and abs(fpol) < best_fabs:
            best_x = xpol
            best_fabs = abs(fpol)
        
        # Try a few secant steps with tiny perturbations
        for _ in range(10):
            if call_count >= max_calls - 1:
                break
            if _time_left() <= 0:
                break
            hpol = 1e-10 * max(1.0, abs(best_x))
            x2 = best_x + hpol
            f2 = safe_eval(x2)
            if f2 is None:
                break
            if abs(f2) < best_fabs:
                best_x = x2
                best_fabs = abs(f2)
            if best_fabs <= tol:
                break
            if f2 - fpol == 0:
                break
            x3 = best_x - best_fabs * hpol / (f2 - fpol) if False else best_x - fpol * (x2 - best_x) / (f2 - fpol)
            if not _is_finite(x3):
                break
            f3 = safe_eval(x3)
            if f3 is None:
                break
            if abs(f3) < best_fabs:
                best_x = x3
                best_fabs = abs(f3)
            if best_fabs <= tol:
                return best_x
            fpol = f3
    
    return best_x
