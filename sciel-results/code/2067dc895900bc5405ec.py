"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if math.isfinite(val) and abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    def is_good(x, fx):
        return abs(fx) <= TOL
    
    def newton_raphson(x0):
        nonlocal evals, best_x, best_abs
        x = float(x0)
        for _ in range(30):
            if time.monotonic() > DEADLINE:
                break
            fx = call_f(x)
            if fx is None:
                break
            if is_good(x, fx):
                return x
            # Use central difference for better accuracy
            h = 1e-5 * max(1.0, abs(x))
            fx_plus = call_f(x + h)
            fx_minus = call_f(x - h)
            if fx_plus is None or fx_minus is None:
                break
            d = (fx_plus - fx_minus) / (2 * h)
            if d == 0 or not math.isfinite(d):
                break
            step = fx / d
            # Damping: if step is too large, reduce it
            new_x = x - step
            if not math.isfinite(new_x):
                break
            # Check for overflow/divergence
            if abs(new_x - x) > 10 * max(1.0, abs(x)):
                new_x = x - math.copysign(10 * max(1.0, abs(x)) * 0.5, step)
            x = new_x
        return None
    
    def secant_method(x0):
        nonlocal evals, best_x, best_abs
        x0 = float(x0)
        h = 1e-4 * max(1.0, abs(x0))
        x1 = x0 + h
        fx0 = call_f(x0)
        fx1 = call_f(x1)
        if fx0 is None or fx1 is None:
            return None
        if is_good(x0, fx0):
            return x0
        if is_good(x1, fx1):
            return x1
        
        for _ in range(40):
            if time.monotonic() > DEADLINE:
                break
            if fx0 == fx1:
                break
            x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
            if not math.isfinite(x2):
                break
            fx2 = call_f(x2)
            if fx2 is None:
                break
            if is_good(x2, fx2):
                return x2
            x0, fx0 = x1, fx1
            x1, fx1 = x2, fx2
        return None
    
    # Try Newton first
    result = newton_raphson(x0)
    if result is not None:
        return result
    
    # If Newton failed, try secant
    result = secant_method(x0)
    if result is not None:
        return result
    
    # Return the best x found
    return best_x
# EVOLVE-BLOCK-END
