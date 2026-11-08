"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f near x0 using a robust combination of methods:
    1. Try Newton's method with adaptive step size and damping
    2. If that fails, try secant method
    3. If that fails, try bisection on a bracketed interval
    4. Track the best solution found (smallest |f(x)|)
    
    Returns x with |f(x)| <= 1e-8 if possible, within 500 evaluations and 150s.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Wrap f to count evaluations and handle potential issues
    eval_count = 0
    max_evals = 500
    
    def safe_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return None
        eval_count += 1
        try:
            val = f(x)
            if val is None:
                return None
            val = float(val)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError, TypeError):
            return None
    
    def time_ok():
        return time.time() < deadline and eval_count < max_evals
    
    x = float(x0)
    
    # First evaluation
    fx = safe_f(x)
    if fx is None:
        return x
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)
    
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Strategy 1: Newton's method with adaptive finite difference and damping
    h_base = 1e-5 * max(1.0, abs(x))
    x_newton = x
    fx_newton = fx
    newton_success = False
    
    for i in range(100):
        if not time_ok():
            break
        
        # Finite difference derivative with adaptive step
        h = h_base * max(1.0, abs(x_newton))
        fx_plus = safe_f(x_newton + h)
        if fx_plus is None:
            break
        
        # Try central difference if we have budget
        if eval_count < max_evals - 1:
            fx_minus = safe_f(x_newton - h)
            if fx_minus is not None:
                d = (fx_plus - fx_minus) / (2 * h)
            else:
                d = (fx_plus - fx_newton) / h
        else:
            d = (fx_plus - fx_newton) / h
        
        if d == 0 or math.isnan(d) or math.isinf(d):
            break
        
        # Newton step
        step = fx_newton / d
        
        # Damping: if step is too large, reduce it
        max_step = 1.0 * max(1.0, abs(x_newton))
        if abs(step) > max_step:
            step = max_step * math.copysign(1.0, step)
        
        x_trial = x_newton - step
        fx_trial = safe_f(x_trial)
        
        if fx_trial is None:
            # Try smaller step
            step *= 0.5
            x_trial = x_newton - step
            fx_trial = safe_f(x_trial)
            if fx_trial is None:
                break
        
        # Update best
        abs_fx_trial = abs(fx_trial)
        if abs_fx_trial < best_abs_fx:
            best_x = x_trial
            best_fx = fx_trial
            best_abs_fx = abs_fx_trial
        
        if best_abs_fx <= 1e-8:
            return best_x
        
        # Accept if it reduces |f|, or if it's a small step
        if abs_fx_trial < abs(fx_newton) * 0.5 or abs(step) < 1e-10:
            x_newton = x_trial
            fx_newton = fx_trial
        else:
            # Damp more
            step *= 0.25
            x_trial = x_newton - step
            fx_trial = safe_f(x_trial)
            if fx_trial is not None:
                abs_fx_trial = abs(fx_trial)
                if abs_fx_trial < best_abs_fx:
                    best_x = x_trial
                    best_fx = fx_trial
                    best_abs_fx = abs_fx_trial
                if best_abs_fx <= 1e-8:
                    return best_x
                if abs_fx_trial < abs(fx_newton):
                    x_newton = x_trial
                    fx_newton = fx_trial
                else:
                    break
            else:
                break
    
    # Strategy 2: Secant method
    if best_abs_fx > 1e-8 and time_ok():
        x0_sec = x
        x1_sec = x + 1e-4 * max(1.0, abs(x))
        fx0 = safe_f(x0_sec)
        fx1 = safe_f(x1_sec)
        
        if fx0 is not None and fx1 is not None:
            for i in range(100):
                if not time_ok():
                    break
                
                if fx1 == 0 or abs(fx1) <= 1e-8:
                    best_x = x1_sec
                    best_fx = fx1
                    best_abs_fx = abs(fx1)
                    break
                
                if fx0 == 0:
                    best_x = x0_sec
                    best_fx = fx0
                    best_abs_fx = abs(fx0)
                    break
                
                denom = fx1 - fx0
                if denom == 0 or math.isnan(denom) or math.isinf(denom):
                    break
                
                x2 = x1_sec - fx1 * (x1_sec - x0_sec) / denom
                fx2 = safe_f(x2)
                
                if fx2 is None:
                    break
                
                abs_fx2 = abs(fx2)
                if abs_fx2 < best_abs_fx:
                    best_x = x2
                    best_fx = fx2
                    best_abs_fx = abs_fx2
                
                if best_abs_fx <= 1e-8:
                    break
                
                # Check for sign change to switch to bisection
                if fx0 * fx2 < 0:
                    # We have a bracket
                    a, fa = x0_sec, fx0
                    b, fb = x2, fx2
                    for j in range(50):
                        if not time_ok():
                            break
                        m = 0.5 * (a + b)
                        fm = safe_f(m)
                        if fm is None:
                            break
                        abs_fm = abs(fm)
                        if abs_fm < best_abs_fx:
                            best_x = m
                            best_fx = fm
                            best_abs_fx = abs_fm
                        if best_abs_fx <= 1e-8:
                            break
                        if fa * fm <= 0:
                            b = m
                            fb = fm
                        else:
                            a = m
                            fa = fm
                        if abs(b - a) < 1e-12 * max(1.0, abs(m)):
                            break
                    break
                elif fx1 * fx2 < 0:
                    a, fa = x1_sec, fx1
                    b, fb = x2, fx2
                    for j in range(50):
                        if not time_ok():
                            break
                        m = 0.5 * (a + b)
                        fm = safe_f(m)
                        if fm is None:
                            break
                        abs_fm = abs(fm)
                        if abs_fm < best_abs_fx:
                            best_x = m
                            best_fx = fm
                            best_abs_fx = abs_fm
                        if best_abs_fx <= 1e-8:
                            break
                        if fa * fm <= 0:
                            b = m
                            fb = fm
                        else:
                            a = m
                            fa = fm
                        if abs(b - a) < 1e-12 * max(1.0, abs(m)):
                            break
                    break
                
                x0_sec, fx0 = x1_sec, fx1
                x1_sec, fx1 = x2, fx2
    
    # Strategy 3: Try to find a bracket and use bisection
    if best_abs_fx > 1e-8 and time_ok():
        # Try expanding from x0 to find a sign change
        a = x0
        fa = safe_f(a)
        if fa is None:
            return best_x
        
        step_size = 1e-4 * max(1.0, abs(a))
        for direction in [1, -1]:
            b = a + direction * step_size
            fb = safe_f(b)
            if fb is None:
                continue
            
            if fa * fb < 0:
                # Found a bracket, use bisection
                for i in range(80):
                    if not time_ok():
                        break
                    m = 0.5 * (a + b)
                    fm = safe_f(m)
                    if fm is None:
                        break
                    abs_fm = abs(fm)
                    if abs_fm < best_abs_fx:
                        best_x = m
                        best_fx = fm
                        best_abs_fx = abs_fm
                    if best_abs_fx <= 1e-8:
                        break
                    if fa * fm <= 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                    if abs(b - a) < 1e-12 * max(1.0, abs(m)):
                        break
                break
            else:
                # Expand
                a = b
                fa = fb
                step_size *= 2
                if step_size > 1e6 * max(1.0, abs(x0)):
                    break
    
    # Final check
    if best_abs_fx <= 1e-8:
        return best_x
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
