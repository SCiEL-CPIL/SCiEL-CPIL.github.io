"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned functions.
    Uses Newton with damping, secant fallback, and bisection.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 120s limit
    eval_count = 0
    MAX_EVALS = 500
    
    def check_deadline():
        return time.time() > deadline or eval_count >= MAX_EVALS
    
    # Track best solution
    best_x = float(x0)
    best_fx = f(best_x)
    eval_count += 1
    best_abs = abs(best_fx)
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 1: Newton's method with damping
    x = float(x0)
    fx = best_fx
    for i in range(25):
        if check_deadline():
            break
        if abs(fx) <= 1e-8:
            return x
        
        # Compute derivative via central difference
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = f(x + h)
        fx_minus = f(x - h)
        eval_count += 2
        if check_deadline():
            break
        dfx = (fx_plus - fx_minus) / (2 * h)
        
        if abs(dfx) < 1e-15:
            break  # derivative too small, switch to secant
        
        # Newton step with damping
        step = fx / dfx
        # Limit step size
        max_step = 1.0 + abs(x)
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)
        
        x_new = x - step
        fx_new = f(x_new)
        eval_count += 1
        if check_deadline():
            break
        
        # Update best
        if abs(fx_new) < best_abs:
            best_x = x_new
            best_fx = fx_new
            best_abs = abs(fx_new)
            if best_abs <= 1e-8:
                return best_x
        
        # Damping: if Newton step doesn't reduce |f|, try smaller step
        if abs(fx_new) > abs(fx):
            # Try damped step
            for alpha in [0.5, 0.25, 0.1, 0.05]:
                x_damped = x - alpha * step
                fx_damped = f(x_damped)
                eval_count += 1
                if check_deadline():
                    break
                if abs(fx_damped) < best_abs:
                    best_x = x_damped
                    best_fx = fx_damped
                    best_abs = abs(fx_damped)
                    if best_abs <= 1e-8:
                        return best_x
                if abs(fx_damped) < abs(fx):
                    x = x_damped
                    fx = fx_damped
                    break
            else:
                # No improvement, break to secant
                break
        else:
            x = x_new
            fx = fx_new
    
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 2: Secant method
    if not check_deadline():
        x0_s = best_x
        x1_s = x0_s + 1e-4 * max(1.0, abs(x0_s))
        f0_s = best_fx
        f1_s = f(x1_s)
        eval_count += 1
        if check_deadline():
            return best_x if best_abs <= 1e-8 else best_x
        
        for i in range(25):
            if check_deadline():
                break
            if abs(f1_s) < best_abs:
                best_x = x1_s
                best_fx = f1_s
                best_abs = abs(f1_s)
                if best_abs <= 1e-8:
                    return best_x
            
            if abs(f0_s) < best_abs:
                best_x = x0_s
                best_fx = f0_s
                best_abs = abs(f0_s)
                if best_abs <= 1e-8:
                    return best_x
            
            if abs(f1_s - f0_s) < 1e-15:
                break
            
            # Secant step
            x2_s = x1_s - f1_s * (x1_s - x0_s) / (f1_s - f0_s)
            # Limit step
            if abs(x2_s - x1_s) > 2.0 * max(1.0, abs(x1_s)):
                x2_s = x1_s + 2.0 * max(1.0, abs(x1_s)) * (1.0 if x2_s > x1_s else -1.0)
            
            f2_s = f(x2_s)
            eval_count += 1
            if check_deadline():
                break
            
            if abs(f2_s) < best_abs:
                best_x = x2_s
                best_fx = f2_s
                best_abs = abs(f2_s)
                if best_abs <= 1e-8:
                    return best_x
            
            x0_s = x1_s
            f0_s = f1_s
            x1_s = x2_s
            f1_s = f2_s
    
    if best_abs <= 1e-8:
        return best_x
    
    # Phase 3: Bisection with bracket expansion
    if not check_deadline():
        # Try to find a bracket around best_x
        a = best_x - 1.0
        b = best_x + 1.0
        fa = f(a)
        fb = f(b)
        eval_count += 2
        if check_deadline():
            return best_x
        
        # Expand bracket if no sign change
        scale = 1.0
        while fa * fb > 0 and scale < 1e6 and not check_deadline():
            scale *= 2.0
            a = best_x - scale
            b = best_x + scale
            fa = f(a)
            fb = f(b)
            eval_count += 2
            if check_deadline():
                break
        
        # Bisection
        if fa * fb <= 0 and not check_deadline():
            for i in range(60):
                if check_deadline():
                    break
                mid = (a + b) / 2.0
                fm = f(mid)
                eval_count += 1
                if check_deadline():
                    break
                
                if abs(fm) < best_abs:
                    best_x = mid
                    best_fx = fm
                    best_abs = abs(fm)
                    if best_abs <= 1e-8:
                        return best_x
                
                if fm == 0:
                    return mid
                if fa * fm <= 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
    
    return best_x
# EVOLVE-BLOCK-END
