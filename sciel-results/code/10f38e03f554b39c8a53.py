"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Robust root finder for ill-conditioned problems.
    Strategy:
    1. Try Newton with adaptive damping (fast when derivative is good)
    2. If Newton fails, try secant method
    3. If secant fails, bracket and use safeguarded secant/bisection
    4. Final high-precision polish pass if close to root
    5. Deadline guard to stay within time limits
    """
    deadline = time.time() + 145  # Leave 5s margin before 150s kill
    max_evals = 480
    eval_count = [0]
    best_x = float(x0)
    best_fx = None
    best_abs_fx = float('inf')
    
    def eval_f(x):
        """Evaluate f with counting and deadline check."""
        nonlocal best_x, best_fx, best_abs_fx
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        try:
            val = f(x)
            eval_count[0] += 1
            if val is None or math.isnan(val) or math.isinf(val):
                return None
            abs_val = abs(val)
            if abs_val < best_abs_fx:
                best_abs_fx = abs_val
                best_x = x
                best_fx = val
            return val
        except (ValueError, OverflowError, ZeroDivisionError, FloatingPointError):
            return None
    
    def is_good(x, fx):
        return fx is not None and not math.isnan(fx) and not math.isinf(fx) and abs(fx) <= 1e-8
    
    def polish(x_start, fx_start, max_iters=20):
        """
        High-precision polish pass using secant/Newton hybrid.
        Only called when |f(x)| < 1e-6 to save evaluations.
        """
        if fx_start is None or math.isnan(fx_start) or math.isinf(fx_start):
            return x_start, fx_start
        if abs(fx_start) >= 1e-6:
            return x_start, fx_start
        
        x = float(x_start)
        fx = float(fx_start)
        x_prev = x + 1e-8 * max(1.0, abs(x))
        fx_prev = None
        
        for i in range(max_iters):
            if time.time() > deadline:
                break
            if eval_count[0] >= max_evals - 5:
                break
            
            # Try to get derivative via finite difference if we have previous point
            if fx_prev is not None and x_prev != x:
                d = (fx - fx_prev) / (x - x_prev)
                if d != 0 and not math.isnan(d) and not math.isinf(d):
                    step = fx / d
                    # Damp if step is too large
                    if abs(step) > 0.5 * max(1e-10, abs(x)):
                        step = math.copysign(0.5 * max(1e-10, abs(x)), step)
                    x_new = x - step
                    if not (math.isnan(x_new) or math.isinf(x_new)):
                        fx_new = eval_f(x_new)
                        if fx_new is None:
                            break
                        if is_good(x_new, fx_new):
                            return x_new, fx_new
                        x_prev = x
                        fx_prev = fx
                        x = x_new
                        fx = fx_new
                        continue
            
            # Fallback: secant step
            if fx_prev is not None and x_prev != x and fx != fx_prev:
                x_new = x - fx * (x - x_prev) / (fx - fx_prev)
                if not (math.isnan(x_new) or math.isinf(x_new)):
                    fx_new = eval_f(x_new)
                    if fx_new is None:
                        break
                    if is_good(x_new, fx_new):
                        return x_new, fx_new
                    x_prev = x
                    fx_prev = fx
                    x = x_new
                    fx = fx_new
                    continue
            
            # If we can't make progress, break
            break
        
        return x, fx
    
    # Phase 1: Try Newton's method with adaptive damping
    x = float(x0)
    for i in range(30):
        if time.time() > deadline:
            break
        fx = eval_f(x)
        if fx is None:
            break
        if is_good(x, fx):
            return x
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        fx_plus = eval_f(x + h)
        if fx_plus is None:
            break
        d = (fx_plus - fx) / h
        if d == 0 or math.isnan(d) or math.isinf(d):
            break
        # Newton step with damping
        step = fx / d
        # Damp if step is too large
        if abs(step) > 10 * max(1.0, abs(x)):
            step = math.copysign(10 * max(1.0, abs(x)), step)
        x_new = x - step
        if math.isnan(x_new) or math.isinf(x_new):
            break
        x = x_new
    
    if is_good(best_x, best_fx):
        return best_x
    
    # Phase 1b: Polish if close to root
    if best_abs_fx < 1e-6:
        x_polished, fx_polished = polish(best_x, best_fx)
        if is_good(x_polished, fx_polished):
            return x_polished
    
    # Phase 2: Try secant method
    x0_sec = float(x0)
    x1_sec = x0_sec + 1e-4 * max(1.0, abs(x0_sec))
    fx0 = eval_f(x0_sec)
    fx1 = eval_f(x1_sec)
    if fx0 is not None and is_good(x0_sec, fx0):
        return x0_sec
    if fx1 is not None and is_good(x1_sec, fx1):
        return x1_sec
    
    for i in range(40):
        if time.time() > deadline:
            break
        if fx0 is None or fx1 is None:
            break
        if math.isnan(fx0) or math.isinf(fx0) or math.isnan(fx1) or math.isinf(fx1):
            break
        if fx0 == fx1:
            break
        x2 = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
        if math.isnan(x2) or math.isinf(x2):
            break
        fx2 = eval_f(x2)
        if fx2 is None:
            break
        if is_good(x2, fx2):
            return x2
        x0_sec = x1_sec
        fx0 = fx1
        x1_sec = x2
        fx1 = fx2
    
    if is_good(best_x, best_fx):
        return best_x
    
    # Phase 2b: Polish if close to root
    if best_abs_fx < 1e-6:
        x_polished, fx_polished = polish(best_x, best_fx)
        if is_good(x_polished, fx_polished):
            return x_polished
    
    # Phase 3: Bracket and use safeguarded secant/bisection
    # Try to find a bracket around x0
    a = float(x0)
    fa = eval_f(a)
    if fa is None:
        return best_x
    if is_good(a, fa):
        return a
    
    # Expand bracket
    step_size = 1e-4 * max(1.0, abs(a))
    for i in range(50):
        if time.time() > deadline:
            break
        # Try expanding in both directions
        b = a + step_size * (i + 1)
        fb = eval_f(b)
        if fb is not None and not math.isnan(fb) and not math.isinf(fb):
            if is_good(b, fb):
                return b
            if fa * fb < 0:
                break
        b = a - step_size * (i + 1)
        fb = eval_f(b)
        if fb is not None and not math.isnan(fb) and not math.isinf(fb):
            if is_good(b, fb):
                return b
            if fa * fb < 0:
                break
    
    # If we have a bracket, use safeguarded secant
    # Find a and b with sign change
    a = float(x0)
    fa = eval_f(a)
    if fa is None:
        return best_x
    if is_good(a, fa):
        return a
    
    # Try multiple expansion strategies
    bracket_found = False
    for scale in [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        if time.time() > deadline:
            break
        step = scale * max(1.0, abs(a))
        for direction in [1, -1]:
            if time.time() > deadline:
                break
            b = a + direction * step
            fb = eval_f(b)
            if fb is None or math.isnan(fb) or math.isinf(fb):
                continue
            if is_good(b, fb):
                return b
            if fa * fb < 0:
                bracket_found = True
                break
        if bracket_found:
            break
    
    if bracket_found and fa is not None and fb is not None:
        # Safeguarded secant method
        for i in range(200):
            if time.time() > deadline:
                break
            if eval_count[0] >= max_evals:
                break
            # Secant step
            if fa == fb:
                # Bisection
                x_mid = (a + b) / 2
            else:
                x_mid = b - fb * (b - a) / (fb - fa)
            
            # Check if x_mid is within bracket
            if not (min(a, b) < x_mid < max(a, b)):
                x_mid = (a + b) / 2
            
            f_mid = eval_f(x_mid)
            if f_mid is None:
                break
            if is_good(x_mid, f_mid):
                return x_mid
            if math.isnan(f_mid) or math.isinf(f_mid):
                # Use bisection
                x_mid = (a + b) / 2
                f_mid = eval_f(x_mid)
                if f_mid is None:
                    break
                if is_good(x_mid, f_mid):
                    return x_mid
                if math.isnan(f_mid) or math.isinf(f_mid):
                    break
            
            if fa * f_mid <= 0:
                b = x_mid
                fb = f_mid
            else:
                a = x_mid
                fa = f_mid
            
            if abs(b - a) < 1e-12 * max(1.0, abs(x_mid)):
                break
        
        # Return the point with smallest |f|
        if abs(fa) <= abs(fb):
            return a
        else:
            return b
    
    # Final polish attempt before returning
    if best_abs_fx < 1e-6:
        x_polished, fx_polished = polish(best_x, best_fx)
        if is_good(x_polished, fx_polished):
            return x_polished
    
    return best_x
# EVOLVE-BLOCK-END
