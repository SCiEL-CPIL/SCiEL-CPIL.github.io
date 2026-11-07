"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from scipy.optimize import brentq

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
    
    # First, try to find a bracket via geometric expansion
    # This is crucial for robustness with polynomial roots
    lo, hi, flo, fhi = None, None, None, None
    step = max(1e-4, abs(x) * 1e-4)
    
    for i in range(60):
        if not time_ok():
            break
        x_left = x - step
        x_right = x + step
        
        try:
            f_left = f(x_left)
        except Exception:
            f_left = None
        
        try:
            f_right = f(x_right)
        except Exception:
            f_right = None
        
        if f_left is not None and math.isfinite(f_left):
            if abs(f_left) < best_abs_fx:
                best_x, best_fx, best_abs_fx = x_left, f_left, abs(f_left)
            if abs(f_left) <= 1e-8:
                return x_left
            if flo is not None and flo * f_left < 0:
                # Sign change between lo and x_left
                lo, hi = x_left, lo
                flo, fhi = f_left, flo
                break
            if f_right is not None and math.isfinite(f_right) and f_left * f_right < 0:
                # Sign change between x_left and x_right
                lo, hi = x_left, x_right
                flo, fhi = f_left, f_right
                break
            if flo is None:
                lo, flo = x_left, f_left
        
        if f_right is not None and math.isfinite(f_right):
            if abs(f_right) < best_abs_fx:
                best_x, best_fx, best_abs_fx = x_right, f_right, abs(f_right)
            if abs(f_right) <= 1e-8:
                return x_right
            if fhi is not None and fhi * f_right < 0:
                # Sign change between hi and x_right
                lo, hi = hi, x_right
                flo, fhi = fhi, f_right
                break
            if fhi is None:
                hi, fhi = x_right, f_right
        
        # Double the step size
        step *= 2.0
    
    # If we found a bracket, use Brent's method
    if lo is not None and hi is not None and flo is not None and fhi is not None:
        if flo * fhi < 0:
            try:
                root = brentq(f, lo, hi, xtol=1e-15, rtol=1e-15, maxiter=100)
                if math.isfinite(root):
                    f_root = f(root)
                    if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                        return root
                    # Even if not quite 1e-8, it's likely better
                    if abs(f_root) < best_abs_fx:
                        return root
                    return root
            except Exception:
                pass
    
    # If no bracket or Brent failed, use safeguarded Newton-Raphson
    # Approximate derivative using central difference
    h = 1e-8
    try:
        f_plus = f(x + h)
        f_minus = f(x - h)
        if math.isfinite(f_plus) and math.isfinite(f_minus):
            fp = (f_plus - f_minus) / (2 * h)
        else:
            fp = None
    except Exception:
        fp = None
    
    if fp is None or abs(fp) < 1e-15:
        # Try forward difference
        try:
            f_plus = f(x + h)
            if math.isfinite(f_plus):
                fp = (f_plus - fx) / h
            else:
                fp = None
        except Exception:
            fp = None
    
    if fp is None or abs(fp) < 1e-15:
        # Fallback to secant method if derivative is problematic
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
        x1, f1 = x, fx
        for i in range(80):
            if not time_ok():
                break
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            if abs(f2 - f1) < 1e-15:
                x_new = x2 + 1e-4
            else:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            max_step = max(1.0, abs(x2) * 0.5)
            if abs(x_new - x2) > max_step:
                x_new = x2 + math.copysign(max_step, x_new - x2)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
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
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return best_x
    
    # Newton-Raphson with safeguards
    for i in range(100):
        if not time_ok():
            break
        
        # Compute Newton step
        step = fx / fp
        x_new = x - step
        
        # Safeguard: if step is too large, dampen it
        max_step = max(1.0, abs(x) * 0.5)
        if abs(step) > max_step:
            step = math.copysign(max_step, step)
            x_new = x - step
        
        # Try the Newton step
        try:
            f_new = f(x_new)
        except Exception:
            # If Newton step fails, try a smaller step
            x_new = x - 0.5 * step
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        
        if not math.isfinite(f_new):
            # Try even smaller step
            x_new = x - 0.25 * step
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
        
        # Check if we made progress
        if abs(f_new) < abs(fx) * 0.9:
            # Good progress, accept the step
            x, fx = x_new, f_new
            # Recompute derivative
            try:
                f_plus = f(x + h)
                f_minus = f(x - h)
                if math.isfinite(f_plus) and math.isfinite(f_minus):
                    fp = (f_plus - f_minus) / (2 * h)
                else:
                    fp = None
            except Exception:
                fp = None
            if fp is None or abs(fp) < 1e-15:
                try:
                    f_plus = f(x + h)
                    if math.isfinite(f_plus):
                        fp = (f_plus - fx) / h
                    else:
                        fp = None
                except Exception:
                    fp = None
            if fp is None or abs(fp) < 1e-15:
                # Derivative failed, fall back to secant-like step
                break
        else:
            # No good progress, try to find a bracket or use bisection
            found_bracket = False
            for dx in [abs(step) * 0.5, abs(step) * 0.25, abs(step) * 0.1, 1e-3, 1e-2, 0.1, 1.0]:
                for sign in [1, -1]:
                    x_test = x + sign * dx
                    try:
                        f_test = f(x_test)
                    except Exception:
                        continue
                    if not math.isfinite(f_test):
                        continue
                    if abs(f_test) < best_abs_fx:
                        best_x = x_test
                        best_fx = f_test
                        best_abs_fx = abs(f_test)
                    if abs(f_test) <= 1e-8:
                        return x_test
                    if fx * f_test < 0:
                        # Found a bracket, use bisection
                        lo_b, hi_b = (x, x_test) if sign == 1 else (x_test, x)
                        flo_b, fhi_b = (fx, f_test) if sign == 1 else (f_test, fx)
                        for j in range(50):
                            if not time_ok():
                                break
                            if abs(flo_b) <= 1e-8:
                                return lo_b
                            if abs(fhi_b) <= 1e-8:
                                return hi_b
                            mid = 0.5 * (lo_b + hi_b)
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
                            if flo_b * fmid <= 0:
                                hi_b, fhi_b = mid, fmid
                            else:
                                lo_b, flo_b = mid, fmid
                        return best_x
            # No bracket found, take a small step in the direction that decreases |f|
            for dx in [1e-3, 1e-2, 0.1, 1.0]:
                for sign in [1, -1]:
                    x_test = x + sign * dx
                    try:
                        f_test = f(x_test)
                    except Exception:
                        continue
                    if not math.isfinite(f_test):
                        continue
                    if abs(f_test) < abs(fx):
                        x, fx = x_test, f_test
                        if abs(fx) < best_abs_fx:
                            best_x = x
                            best_fx = fx
                            best_abs_fx = abs(fx)
                        if abs(fx) <= 1e-8:
                            return x
                        found_bracket = True
                        break
                if found_bracket:
                    break
            if not found_bracket:
                break
    
    return best_x
# EVOLVE-BLOCK-END
