"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using Brent's method (adaptive IQI + bisection).
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
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
    
    # Phase 1: Find a bracket efficiently
    a, b, fa, fb = _find_bracket(f, x0, call_f, check_time, update_best)
    
    if a is not None and b is not None:
        # Phase 2: Brent's method (IQI + bisection hybrid)
        try:
            result = _brent_method(f, a, b, fa, fb, call_f, check_time, update_best)
            if result is not None:
                return result
        except Exception:
            pass
    
    # Fallback: secant from x0
    try:
        result = _secant_fallback(f, x0, call_f, check_time, update_best)
        if result is not None:
            return result
    except Exception:
        pass
    
    # Return best found
    return best_x


def _find_bracket(f, x0, call_f, check_time, update_best):
    """Find a bracket [a,b] with f(a)*f(b) <= 0 using minimal evaluations."""
    x = float(x0)
    try:
        fx = call_f(x)
    except Exception:
        return None, None, None, None
    update_best(x, fx)
    
    if fx is None:
        return None, None, None, None
    if abs(fx) <= 1e-8:
        return x, x, fx, fx
    
    # Try multiple step sizes: small to large, both directions
    base_h = max(1e-4, abs(x) * 1e-4)
    
    for i in range(40):
        if not check_time():
            return None, None, None, None
        
        h = base_h * (2.0 ** i)
        
        # Try x + h
        x_plus = x + h
        try:
            fp = call_f(x_plus)
        except Exception:
            fp = None
        update_best(x_plus, fp)
        
        if fp is not None and fp * fx <= 0:
            if abs(x_plus - x) > 0:
                return x, x_plus, fx, fp
        
        # Try x - h
        x_minus = x - h
        try:
            fm = call_f(x_minus)
        except Exception:
            fm = None
        update_best(x_minus, fm)
        
        if fm is not None and fm * fx <= 0:
            if abs(x_minus - x) > 0:
                return x_minus, x, fm, fx
    
    return None, None, None, None


def _brent_method(f, a, b, fa, fb, call_f, check_time, update_best):
    """
    Brent's method: combines bisection with inverse quadratic interpolation.
    Maintains a bracket [a,b] where f(a)*f(b) <= 0 throughout.
    """
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb > 0:
        return None
    
    # Standard Brent's method setup
    # a, b: bracket endpoints with f(a)*f(b) <= 0
    # c: previous value of b (or a)
    c = b
    fc = fb
    
    # Ensure |f(a)| <= |f(b)| so a is the "better" endpoint
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        c = b
        fc = fb
    
    tol = 1e-14
    max_iter = 200
    muel = 0.5 * (b - a)  # half the interval
    d = muel
    
    for i in range(max_iter):
        if not check_time():
            return None
        
        # Check convergence
        if abs(fa) <= 1e-8:
            return a
        if abs(b - a) <= tol * max(1.0, abs(a)):
            return a if abs(fa) < abs(fb) else b
        
        # Try inverse quadratic interpolation
        used_iqi = False
        x_iqi = None
        
        if a != b and b != c and a != c:
            try:
                if abs(fa - fb) > 1e-15 and abs(fb - fc) > 1e-15 and abs(fa - fc) > 1e-15:
                    term1 = fc * (c - a) / (fc - fa) * (c - b) / (fc - fb)
                    term2 = fa * (c - a) / (fa - fc) * (a - b) / (fa - fb)
                    term3 = fb * (c - b) / (fb - fc) * (b - a) / (fb - fa)
                    x_iqi = c + term1 + term2 + term3
                    
                    # Check if x_iqi is within the bracket
                    lo = min(a, b)
                    hi = max(a, b)
                    if lo < x_iqi < hi:
                        used_iqi = True
            except (ZeroDivisionError, OverflowError, ValueError):
                pass
        
        if not used_iqi:
            # Bisection
            x_iqi = 0.5 * (a + b)
        
        # Evaluate
        try:
            f_iqi = call_f(x_iqi)
        except Exception:
            break
        update_best(x_iqi, f_iqi)
        
        if f_iqi is None:
            break
        if abs(f_iqi) <= 1e-8:
            return x_iqi
        
        # Update bracket: maintain f(a)*f(b) <= 0
        if fa * f_iqi <= 0:
            # Root is in [a, x_iqi]
            c = b
            fc = fb
            b = x_iqi
            fb = f_iqi
        else:
            # Root is in [x_iqi, b]
            c = b
            fc = fb
            a = x_iqi
            fa = f_iqi
        
        # Ensure |f(a)| <= |f(b)|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        # Update muel and d
        muel = 0.5 * (b - a)
        d = muel
    
    # Return the better endpoint
    if abs(fa) < abs(fb):
        return a
    else:
        return b


def _secant_fallback(f, x0, call_f, check_time, update_best):
    """Simple secant method as fallback."""
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
            break
        
        if not math.isfinite(x_new):
            break
        
        step = x_new - x
        if abs(step) > 10 * max(1.0, abs(x)):
            step = 10 * max(1.0, abs(x)) * math.copysign(1, step)
        
        x_prev, x = x, x + step
        try:
            fx = call_f(x)
        except Exception:
            break
        update_best(x, fx)
        
        if fx is None:
            break
        if abs(fx) <= 1e-8:
            return x
    
    return None
# EVOLVE-BLOCK-END
