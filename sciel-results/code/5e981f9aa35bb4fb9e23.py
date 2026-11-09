"""Root finding for ill-conditioned problems with evaluation budget awareness."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline
    
    x0 = float(x0)
    eval_count = 0
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    eval_count += 1
    
    def safe_eval(x):
        nonlocal eval_count
        if time_up():
            return None
        try:
            fx = f(x)
            eval_count += 1
            return fx
        except (ValueError, OverflowError, ZeroDivisionError):
            eval_count += 1
            return None
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if fx is not None:
            ax = abs(fx)
            if ax < best_abs:
                best_x = x
                best_f = fx
                best_abs = ax
            return ax
        return best_abs
    
    # If already good enough
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 1: Secant method (uses 1 eval per step after initial 2)
    # Initialize with x0 and a perturbed point
    h0 = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h0
    f1 = safe_eval(x1)
    
    if f1 is None:
        # Try other perturbation directions
        x1 = x0 - h0
        f1 = safe_eval(x1)
    
    if f1 is not None:
        update_best(x1, f1)
        if best_abs <= 1e-8:
            return best_x
        
        # Secant iteration
        x_prev = x0
        f_prev = best_f
        x_curr = x1
        f_curr = f1
        
        for _ in range(80):
            if time_up():
                break
            if abs(f_curr) <= 1e-12:
                return x_curr
            if abs(f_prev) <= 1e-12:
                return x_prev
            
            denom = f_curr - f_prev
            if abs(denom) < 1e-300:
                break
            
            x_new = x_curr - f_curr * (x_curr - x_prev) / denom
            f_new = safe_eval(x_new)
            
            if f_new is None:
                break
            
            abs_new = update_best(x_new, f_new)
            if abs_new <= 1e-12:
                return x_new
            
            # Check for stagnation
            if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_curr)):
                break
            
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
    
    # Phase 2: Newton with forward-difference derivative (fallback)
    # Use the best point found so far as starting point
    x = best_x
    fx = best_f
    
    for _ in range(50):
        if time_up():
            break
        if abs(fx) <= 1e-12:
            return x
        
        h = 1e-8 * max(1.0, abs(x))
        fx_plus = safe_eval(x + h)
        if fx_plus is None:
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        # Damping if step is too large
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        
        fx_new = safe_eval(x_new)
        if fx_new is None:
            break
        
        abs_new = update_best(x_new, fx_new)
        if abs_new <= 1e-12:
            return x_new
        
        if abs(x_new - x) < 1e-14 * max(1.0, abs(x)):
            break
        
        x = x_new
        fx = fx_new
    
    # Phase 3: Bisection safety net if we have a bracket
    # Try to find a bracket around best_x
    if not time_up():
        # Expand search for sign change
        step = 1e-3 * max(1.0, abs(best_x))
        found_bracket = False
        for trial in range(20):
            if time_up():
                break
            x_left = best_x - step * (trial + 1)
            x_right = best_x + step * (trial + 1)
            f_left = safe_eval(x_left)
            f_right = safe_eval(x_right)
            
            if f_left is None or f_right is None:
                break
            
            update_best(x_left, f_left)
            update_best(x_right, f_right)
            
            if best_abs <= 1e-8:
                return best_x
            
            if f_left * f_right < 0:
                # Found bracket, do bisection
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                for _ in range(60):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            elif f_left * best_f < 0:
                a, b = x_left, best_x
                fa, fb = f_left, best_f
                for _ in range(60):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            elif best_f * f_right < 0:
                a, b = best_x, x_right
                fa, fb = best_f, f_right
                for _ in range(60):
                    if time_up():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_eval(m)
                    if fm is None:
                        break
                    abs_m = update_best(m, fm)
                    if abs_m <= 1e-12:
                        return m
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-14 * max(1.0, abs(m)):
                        break
                break
            else:
                step *= 2
                continue
    
    # Final polish: try a few more Newton steps from best_x with tighter tolerance
    if not time_up() and best_abs > 1e-8:
        x = best_x
        fx = best_f
        for _ in range(20):
            if time_up():
                break
            if abs(fx) <= 1e-10:
                return x
            h = 1e-10 * max(1.0, abs(x))
            fx_plus = safe_eval(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_new = x - fx / d
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
            update_best(x_new, fx_new)
            if best_abs <= 1e-10:
                return best_x
            if abs(x_new - x) < 1e-16 * max(1.0, abs(x)):
                break
            x = x_new
            fx = fx_new
    
    return best_x
# EVOLVE-BLOCK-END
