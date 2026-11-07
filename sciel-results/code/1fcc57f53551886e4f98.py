"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder using a combination of:
    1. Initial bracketing via adaptive sampling
    2. Bisection to guarantee convergence
    3. Newton's method with line search for fast quadratic convergence
    4. Secant method as a derivative-free alternative
    5. Time budget guard
    """
    start_time = time.time()
    time_budget = 10.0  # seconds, well under 120s limit
    
    def time_left():
        return time_budget - (time.time() - start_time)
    
    def is_finite(x):
        return math.isfinite(x)
    
    # Track best x found so far
    best_x = float(x0)
    best_fx = f(best_x)
    best_abs = abs(best_fx)
    
    def update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if is_finite(x) and is_finite(fx):
            afx = abs(fx)
            if afx < best_abs:
                best_x = x
                best_fx = fx
                best_abs = afx
    
    update_best(best_x, best_fx)
    
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 1: Try to find a bracket [a, b] where f(a)*f(b) < 0
    # Start with x0 and expand outward
    a = float(x0)
    b = float(x0)
    fa = best_fx
    fb = best_fx
    
    # First, try small steps around x0
    step = 1e-3
    max_expand = 100.0
    
    bracket_found = False
    for i in range(50):
        if time_left() < 0.5:
            break
        
        # Try expanding to the right
        b_new = a + step
        if is_finite(b_new):
            fb_new = f(b_new)
            if is_finite(fb_new):
                update_best(b_new, fb_new)
                if fa * fb_new < 0:
                    a, b, fa, fb = a, b_new, fa, fb_new
                    bracket_found = True
                    break
        
        # Try expanding to the left
        a_new = a - step
        if is_finite(a_new):
            fa_new = f(a_new)
            if is_finite(fa_new):
                update_best(a_new, fa_new)
                if fa_new * fb < 0:
                    a, b, fa, fb = a_new, b, fa_new, fb
                    bracket_found = True
                    break
        
        step *= 2.0
        if step > max_expand:
            break
    
    # If no bracket found, try a wider search
    if not bracket_found:
        # Try a grid search over a reasonable range
        for scale in [1.0, 10.0, 100.0, 1000.0]:
            if time_left() < 0.5:
                break
            for dx in [-scale, -scale/2, -scale/4, scale/4, scale/2, scale]:
                x_try = float(x0) + dx
                if is_finite(x_try):
                    fx_try = f(x_try)
                    if is_finite(fx_try):
                        update_best(x_try, fx_try)
                        if best_abs <= 1e-8:
                            return best_x
                        # Check if we can bracket with x0
                        if fx_try * best_fx < 0:
                            if fx_try < best_fx:
                                a, b, fa, fb = x_try, float(x0), fx_try, best_fx
                            else:
                                a, b, fa, fb = float(x0), x_try, best_fx, fx_try
                            bracket_found = True
                            break
            if bracket_found:
                break
    
    # Phase 2: Use the bracket to converge
    if bracket_found:
        # Bisection + Newton hybrid
        for i in range(100):
            if time_left() < 0.5:
                break
            if best_abs <= 1e-8:
                return best_x
            
            mid = 0.5 * (a + b)
            fm = f(mid)
            if not is_finite(fm):
                break
            update_best(mid, fm)
            
            if abs(fm) <= 1e-8:
                return mid
            
            if fa * fm <= 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            
            # Try Newton step within the bracket
            # Use central difference for derivative
            h = 1e-7 * max(1.0, abs(mid))
            if is_finite(mid + h) and is_finite(mid - h):
                f_plus = f(mid + h)
                f_minus = f(mid - h)
                if is_finite(f_plus) and is_finite(f_minus):
                    d = (f_plus - f_minus) / (2 * h)
                    if is_finite(d) and abs(d) > 1e-15:
                        x_newton = mid - fm / d
                        if is_finite(x_newton) and a < x_newton < b:
                            fx_newton = f(x_newton)
                            if is_finite(fx_newton):
                                update_best(x_newton, fx_newton)
                                if abs(fx_newton) <= 1e-8:
                                    return x_newton
                                # Accept if it reduces |f|
                                if abs(fx_newton) < abs(fm):
                                    if fa * fx_newton <= 0:
                                        b, fb = x_newton, fx_newton
                                    else:
                                        a, fa = x_newton, fx_newton
                                else:
                                    # Revert to bisection
                                    pass
            
            # Check if bracket is small enough
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                break
    else:
        # No bracket found. Use secant/Newton from x0
        # Try Newton with finite difference
        x = float(x0)
        fx = best_fx
        
        for i in range(60):
            if time_left() < 0.5:
                break
            if best_abs <= 1e-8:
                return best_x
            
            h = 1e-7 * max(1.0, abs(x))
            x_plus = x + h
            if not is_finite(x_plus):
                break
            fx_plus = f(x_plus)
            if not is_finite(fx_plus):
                break
            
            d = (fx_plus - fx) / h
            if not is_finite(d) or abs(d) < 1e-15:
                # Try secant with a previous point
                break
            
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            
            fx_new = f(x_new)
            if not is_finite(fx_new):
                break
            
            update_best(x_new, fx_new)
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Damping: if step is too large, shrink it
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
                fx_new = f(x_new)
                if not is_finite(fx_new):
                    break
                update_best(x_new, fx_new)
            
            x, fx = x_new, fx_new
    
    # Final polish: if we have a good candidate, try a few more Newton steps
    if best_abs < 1e-4:
        x = best_x
        fx = best_fx
        for i in range(20):
            if time_left() < 0.2:
                break
            if abs(fx) <= 1e-8:
                return x
            
            h = 1e-8 * max(1.0, abs(x))
            x_plus = x + h
            if not is_finite(x_plus):
                break
            fx_plus = f(x_plus)
            if not is_finite(fx_plus):
                break
            
            d = (fx_plus - fx) / h
            if not is_finite(d) or abs(d) < 1e-15:
                break
            
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            
            fx_new = f(x_new)
            if not is_finite(fx_new):
                break
            
            update_best(x_new, fx_new)
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Only accept if it improves
            if abs(fx_new) >= abs(fx):
                break
            
            x, fx = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
