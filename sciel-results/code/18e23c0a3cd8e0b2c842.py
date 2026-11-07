"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    def time_ok():
        return time.time() < deadline
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Track best point found
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    # Phase 1: Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Start from x0 and expand geometrically in both directions
    a, b = None, None
    fa, fb = None, None
    
    # First, check if x0 itself is close to a sign change
    # Try small perturbations to find a sign change
    step_sizes = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    for h in step_sizes:
        if not time_ok():
            break
        for sign in [1, -1]:
            x_test = x + sign * h
            try:
                fx_test = f(x_test)
            except Exception:
                continue
            if not math.isfinite(fx_test):
                continue
            if abs(fx_test) < best_abs_fx:
                best_x = x_test
                best_fx = fx_test
                best_abs_fx = abs(fx_test)
            if abs(fx_test) <= 1e-8:
                return x_test
            # Check for sign change
            if fx * fx_test < 0:
                if sign == 1:
                    a, b = x, x_test
                    fa, fb = fx, fx_test
                else:
                    a, b = x_test, x
                    fa, fb = fx_test, fx
                break
        if a is not None:
            break
    
    # If we found a bracket, use Brent's method (via bisection + secant hybrid)
    if a is not None and b is not None:
        # Use a robust bracketing method
        lo, hi = a, b
        flo, fhi = fa, fb
        
        for i in range(100):
            if not time_ok():
                break
            
            # Check if we've converged
            if abs(flo) <= 1e-8:
                return lo
            if abs(fhi) <= 1e-8:
                return hi
            
            # Midpoint
            mid = 0.5 * (lo + hi)
            try:
                fmid = f(mid)
            except Exception:
                break
            if not math.isfinite(fmid):
                break
            
            if abs(fmid) < best_abs_fx:
                best_x = mid
                best_fx = fmid
                best_abs_fx = abs(fmid)
            
            if abs(fmid) <= 1e-8:
                return mid
            
            # Decide which half to keep
            if flo * fmid <= 0:
                hi, fhi = mid, fmid
            else:
                lo, flo = mid, fmid
            
            # Also try secant step if it stays in bracket
            if abs(fhi - flo) > 1e-15:
                x_sec = lo - flo * (hi - lo) / (fhi - flo)
                if lo < x_sec < hi:
                    try:
                        f_sec = f(x_sec)
                    except Exception:
                        pass
                    else:
                        if math.isfinite(f_sec):
                            if abs(f_sec) < best_abs_fx:
                                best_x = x_sec
                                best_fx = f_sec
                                best_abs_fx = abs(f_sec)
                            if abs(f_sec) <= 1e-8:
                                return x_sec
                            # Update bracket
                            if flo * f_sec <= 0:
                                hi, fhi = x_sec, f_sec
                            else:
                                lo, flo = x_sec, f_sec
        
        # After bracketing, return best
        if abs(best_fx) <= 1e-8:
            return best_x
        return best_x
    
    # Phase 2: No bracket found. Use secant method with safeguards.
    # Start with two points
    x1 = x
    f1 = fx
    
    # Get a second point
    x2 = x + 1e-3
    try:
        f2 = f(x2)
    except Exception:
        x2 = x - 1e-3
        try:
            f2 = f(x2)
        except Exception:
            return best_x
    if not math.isfinite(f2):
        # Try other offsets
        for dx in [1e-4, 1e-5, 1e-2, 0.1, 1.0]:
            for sign in [1, -1]:
                x_try = x + sign * dx
                try:
                    f_try = f(x_try)
                except Exception:
                    continue
                if math.isfinite(f_try):
                    x2, f2 = x_try, f_try
                    break
            if math.isfinite(f2):
                break
    
    if not math.isfinite(f2):
        return best_x
    
    if abs(f2) < best_abs_fx:
        best_x = x2
        best_fx = f2
        best_abs_fx = abs(f2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Secant method
    for i in range(80):
        if not time_ok():
            break
        
        # Check convergence
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
        
        # Secant step
        if abs(f2 - f1) < 1e-15:
            # Nearly parallel, take a small step
            x_new = x2 + 1e-4
        else:
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        
        # Safeguard: limit step size
        max_step = max(1.0, abs(x2) * 0.5)
        if abs(x_new - x2) > max_step:
            x_new = x2 + math.copysign(max_step, x_new - x2)
        
        try:
            f_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(f_new):
            # Shrink step
            x_new = 0.5 * (x2 + x1)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update: keep the two most recent points
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
    
    return best_x
# EVOLVE-BLOCK-END
