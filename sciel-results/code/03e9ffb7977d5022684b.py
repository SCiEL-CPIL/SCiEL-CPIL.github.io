"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 135.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except Exception:
        return best_x
    
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and (best_fx is None or abs(fx) < abs(best_fx)):
            best_x = x
            best_fx = fx
    
    # Try multiple strategies
    # Strategy 1: Hybrid Newton-Secant with bisection fallback
    try:
        result = _hybrid_newton_secant(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 2: Bisection with bracket expansion
    try:
        result = _bisection_bracket(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 3: Secant method with damping
    try:
        result = _damped_secant(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Strategy 4: Newton with adaptive step
    try:
        result = _adaptive_newton(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Return best found
    return best_x


def _hybrid_newton_secant(f, x0, call_f, check_time, update_best):
    """Hybrid Newton-Secant method with bisection fallback."""
    x = float(x0)
    fx = call_f(x)
    update_best(x, fx)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try to establish a bracket
    a, b = None, None
    fa, fb = None, None
    
    # Expand to find bracket
    h = max(1e-4, abs(x) * 1e-4)
    for i in range(20):
        if not check_time():
            return None
        x_plus = x + h
        x_minus = x - h
        try:
            fp = call_f(x_plus)
            fm = call_f(x_minus)
        except Exception:
            break
        update_best(x_plus, fp)
        update_best(x_minus, fm)
        
        if fp is not None and fm is not None:
            if fp * fx <= 0:
                a, b = x, x_plus
                fa, fb = fx, fp
                break
            elif fm * fx <= 0:
                a, b = x_minus, x
                fa, fb = fm, fx
                break
        h *= 2
    
    # Secant/Newton hybrid
    x_prev = x - 1e-6 * max(1.0, abs(x))
    try:
        f_prev = call_f(x_prev)
    except Exception:
        f_prev = None
    update_best(x_prev, f_prev)
    
    for _ in range(100):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        # If we have a bracket, use bisection
        if a is not None and b is not None and fa is not None and fb is not None:
            if a <= x <= b and fa * fb <= 0:
                # Bisection step
                mid = 0.5 * (a + b)
                try:
                    fmid = call_f(mid)
                except Exception:
                    break
                update_best(mid, fmid)
                if fmid is None:
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid <= 0:
                    b = mid
                    fb = fmid
                else:
                    a = mid
                    fa = fmid
                continue
        
        # Secant step
        if f_prev is not None and x_prev != x:
            dx = x - x_prev
            if abs(dx) > 1e-15:
                x_new = x - fx * dx / (fx - f_prev)
            else:
                # Newton with finite difference
                h = 1e-8 * max(1.0, abs(x))
                try:
                    fxp = call_f(x + h)
                except Exception:
                    break
                if fxp is None:
                    break
                d = (fxp - fx) / h
                if abs(d) < 1e-15:
                    break
                x_new = x - fx / d
        else:
            # Newton with finite difference
            h = 1e-8 * max(1.0, abs(x))
            try:
                fxp = call_f(x + h)
            except Exception:
                break
            if fxp is None:
                break
            d = (fxp - fx) / h
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
        
        # Damping: if step is too large, shrink it
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * max(1.0, abs(x)) * math.copysign(1, x_new - x)
        
        x_prev, x = x, x_new
        f_prev, fx = fx, call_f(x)
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return x
    
    return None


def _bisection_bracket(f, x0, call_f, check_time, update_best):
    """Bisection with bracket expansion from x0."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    # Find bracket
    a, b, fa, fb = None, None, None, None
    h = max(1e-3, abs(x) * 1e-3)
    for i in range(30):
        if not check_time():
            return None
        x_plus = x + h
        x_minus = x - h
        try:
            fp = call_f(x_plus)
            fm = call_f(x_minus)
        except Exception:
            break
        update_best(x_plus, fp)
        update_best(x_minus, fm)
        
        if fp is not None and fm is not None:
            if fp * fx <= 0:
                a, b, fa, fb = x, x_plus, fx, fp
                break
            elif fm * fx <= 0:
                a, b, fa, fb = x_minus, x, fm, fx
                break
        h *= 2.0
    
    if a is None or b is None:
        return None
    
    # Bisection
    for _ in range(100):
        if not check_time():
            return None
        mid = 0.5 * (a + b)
        try:
            fmid = call_f(mid)
        except Exception:
            break
        update_best(mid, fmid)
        if fmid is None:
            break
        if abs(fmid) <= 1e-8:
            return mid
        if fa * fmid <= 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
            break
    
    return None


def _damped_secant(f, x0, call_f, check_time, update_best):
    """Secant method with damping."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x - h
    try:
        f_prev = call_f(x_prev)
    except Exception:
        return None
    update_best(x_prev, f_prev)
    
    if f_prev is None:
        return None
    
    for _ in range(150):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        if f_prev != fx and abs(fx - f_prev) > 1e-15:
            x_new = x - fx * (x - x_prev) / (fx - f_prev)
        else:
            # Fallback to Newton
            h2 = 1e-8 * max(1.0, abs(x))
            try:
                fxp = call_f(x + h2)
            except Exception:
                break
            if fxp is None:
                break
            d = (fxp - fx) / h2
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
        
        # Damping
        step = x_new - x
        if abs(step) > 5 * max(1.0, abs(x)):
            step = 5 * max(1.0, abs(x)) * math.copysign(1, step)
        
        x_prev, x = x, x + step
        f_prev, fx = fx, call_f(x)
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return x
    
    return None


def _adaptive_newton(f, x0, call_f, check_time, update_best):
    """Newton's method with adaptive step size."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None
    update_best(x, fx)
    
    if fx is None:
        return None
    if abs(fx) <= 1e-8:
        return x
    
    step_size = 1.0
    for _ in range(100):
        if not check_time():
            return None
        if abs(fx) <= 1e-8:
            return x
        
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            fxp = call_f(x + h)
        except Exception:
            break
        if fxp is None:
            break
        d = (fxp - fx) / h
        
        if abs(d) < 1e-15:
            break
        
        x_new = x - fx / d
        
        # Adaptive damping
        if not math.isfinite(x_new) or abs(x_new - x) > 10 * max(1.0, abs(x)):
            step_size *= 0.5
            x_new = x - step_size * fx / d
            if not math.isfinite(x_new):
                break
        else:
            step_size = min(1.0, step_size * 1.5)
        
        x = x_new
        fx = call_f(x)
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return x
    
    return None
# EVOLVE-BLOCK-END
