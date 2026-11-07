"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Uses a hybrid strategy:
      1. Try secant method (derivative-free, ~1.5 calls/iter) for smooth polynomials.
      2. If secant diverges or stalls, fall back to bisection with bracketing.
      3. Final Newton polish for high precision.
    Includes a wall-clock deadline guard.
    """
    start_time = time.time()
    DEADLINE = 120.0  # evaluator kills at 120s; use 120 as hard limit, aim to finish well before
    SAFETY_MARGIN = 5.0  # seconds
    
    x = float(x0)
    
    # --- Phase 1: Secant method ---
    # Initialize second point with a small geometric step
    h0 = 1e-4 * max(1.0, abs(x))
    x1 = x + h0
    f0 = f(x)
    f1 = f(x1)
    
    # If we already have a root
    if abs(f0) <= 1e-8:
        return x
    if abs(f1) <= 1e-8:
        return x1
    
    # Secant iteration
    best_x = x
    best_f = f0
    if abs(f1) < abs(best_f):
        best_x = x1
        best_f = f1
    
    max_secant_iters = 30
    for i in range(max_secant_iters):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Check if we have a root
        if abs(f1) <= 1e-10:
            return x1
        if abs(f0) <= 1e-10:
            return x0
        
        # Track best
        if abs(f0) < abs(best_f):
            best_f = f0
            best_x = x0
        if abs(f1) < abs(best_f):
            best_f = f1
            best_x = x1
        
        # Secant step
        df = f1 - f0
        if df == 0.0:
            break
        dx = (x1 - x0) * f1 / df
        x2 = x1 - dx
        
        # Check for non-finite
        if not math.isfinite(x2):
            break
        
        # Shift
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
        
        if not math.isfinite(f1):
            break
    
    if abs(f1) <= 1e-8:
        return x1
    if abs(f0) <= 1e-8:
        return x0
    
    # --- Phase 2: Bracket and bisection if secant failed to converge ---
    # Try to find a bracket around best_x
    if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
        # Return best found
        if abs(best_f) <= 1e-8:
            return best_x
        # Try one more Newton step from best_x
        h = 1e-7 * max(1.0, abs(best_x))
        d = (f(best_x + h) - f(best_x)) / h
        if d != 0 and math.isfinite(d):
            xn = best_x - f(best_x) / d
            if math.isfinite(f(xn)) and abs(f(xn)) <= 1e-8:
                return xn
        return best_x
    
    # Attempt to bracket
    bracket_lo = None
    bracket_hi = None
    f_lo = None
    f_hi = None
    
    # Search outward from best_x
    step = 1e-3 * max(1.0, abs(best_x))
    for attempt in range(20):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        lo = best_x - step
        hi = best_x + step
        f_lo = f(lo)
        f_hi = f(hi)
        
        if not math.isfinite(f_lo) or not math.isfinite(f_hi):
            step *= 2
            continue
        
        if f_lo * f_hi < 0:
            bracket_lo = lo
            bracket_hi = hi
            break
        
        # Also check if either endpoint is a root
        if abs(f_lo) <= 1e-8:
            return lo
        if abs(f_hi) <= 1e-8:
            return hi
        
        # Check if f(best_x) has opposite sign to one endpoint
        if f(best_x) * f_lo < 0:
            bracket_lo = lo
            bracket_hi = best_x
            f_hi = f(best_x)
            break
        if f(best_x) * f_hi < 0:
            bracket_lo = best_x
            bracket_hi = hi
            f_lo = f(best_x)
            break
        
        step *= 2
    
    if bracket_lo is not None:
        # Bisection
        a, b = bracket_lo, bracket_hi
        fa = f(a)
        fb = f(b)
        for _ in range(60):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            mid = 0.5 * (a + b)
            fm = f(mid)
            if abs(fm) <= 1e-10:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        # Return midpoint
        result = 0.5 * (a + b)
        if abs(f(result)) <= 1e-8:
            return result
    
    # --- Phase 3: Final Newton polish from best_x ---
    x = best_x
    fx = f(x)
    for _ in range(20):
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        if abs(fx) <= 1e-12:
            return x
        h = 1e-8 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) >= abs(fx):
            break
        x = x_new
        fx = fx_new
    
    if abs(fx) <= 1e-8:
        return x
    
    # Last resort: return best_x
    if abs(best_f) <= 1e-8:
        return best_x
    
    return x
# EVOLVE-BLOCK-END
