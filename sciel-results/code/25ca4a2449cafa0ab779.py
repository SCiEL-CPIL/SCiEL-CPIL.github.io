import math
import time

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x
    
    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    # Start with small steps, then expand geometrically
    a = x
    fa = fx
    
    # First check nearby points for a sign change
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    bracket_found = False
    b = None
    fb = None
    
    for delta in deltas:
        if time.time() > deadline:
            return x  # Return best so far
        
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break
    
    # If no bracket found with initial deltas, try larger geometric expansion
    if not bracket_found:
        step = 1.0
        for i in range(50):
            if time.time() > deadline:
                return x
            
            # Try expanding in both directions
            for sign in [1, -1]:
                x_try = x + sign * step
                f_try = f(x_try)
                if abs(f_try) <= tol:
                    return x_try
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
            step *= 2.0
    
    if not bracket_found:
        # No bracket found; return the point with smallest |f|
        return x
    
    # Bisection on [a, b]
    # Ensure a < b for clarity (not required for bisection logic)
    lo = a
    hi = b
    flo = fa
    fhi = fb
    
    # Bisection: ~50 iterations to be safe
    for i in range(60):
        if time.time() > deadline:
            # Return midpoint of current bracket
            return (lo + hi) / 2.0
        
        mid = (lo + hi) / 2.0
        fmid = f(mid)
        
        if abs(fmid) <= tol:
            return mid
        
        if flo * fmid < 0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
        
        # If bracket is small enough, we're done
        if abs(hi - lo) < 1e-12:
            break
    
    return (lo + hi) / 2.0
