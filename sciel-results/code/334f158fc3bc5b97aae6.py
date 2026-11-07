import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 400

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x):
    try:
        val = f(float(x))
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        return float('nan')

def _secant_method(f, x0, counter, max_iter=50):
    """
    Robust Secant method with safeguards.
    Uses a larger initial step (0.1) for better-conditioned derivative estimates.
    """
    if counter[0] >= _CALL_BUDGET or _time_left() < 1.0:
        return x0, _safe_call(f, x0), False

    # Initial two points
    x1 = x0
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    if math.isnan(fx1):
        return x1, fx1, False
    if abs(fx1) <= 1e-8:
        return x1, fx1, True

    # Choose x2 carefully.
    # A step of 0.1 is often better conditioned for oscillatory functions than 1e-6.
    # If x0 is very large, scale the step.
    scale = max(1.0, abs(x1))
    # We want a step that is significant relative to the scale, but not too large.
    # 0.1 * scale seems reasonable based on inspiration.
    if x1 == 0.0:
        x2 = 1.0
    else:
        x2 = x1 + 0.1 * scale
    
    fx2 = _safe_call(f, x2)
    counter[0] += 1
    if math.isnan(fx2):
        # If the large step failed, try a smaller one?
        # For now, just return what we have.
        return x2, fx2, False
    if abs(fx2) <= 1e-8:
        return x2, fx2, True

    # Track best point
    best_x = x1 if abs(fx1) < abs(fx2) else x2
    best_fx = fx1 if abs(fx1) < abs(fx2) else fx2

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 1.0:
            break
            
        if abs(fx2) <= 1e-8:
            return x2, fx2, True
        if abs(fx1) <= 1e-8:
            return x1, fx1, True

        # Secant step
        if fx2 == fx1:
            # Degenerate case
            break
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        if not math.isfinite(x_new):
            break
            
        # Safeguard: if the step is huge, it might be diverging.
        # Oscillatory functions can cause large jumps.
        # If |x_new - x2| is very large relative to scale, we might be in trouble.
        # But let's try one step.
        
        x1, fx1 = x2, fx2
        x2 = x_new
        fx2 = _safe_call(f, x2)
        counter[0] += 1
        
        if math.isnan(fx2):
            break
            
        if abs(fx2) < abs(best_fx):
            best_x = x2
            best_fx = fx2

    if abs(best_fx) <= 1e-8:
        return best_x, best_fx, True
        
    return best_x, best_fx, False

def _illinois_method(f, a, b, fa, fb, counter, max_iter=100):
    """
    Illinois variant of Regula Falsi.
    Superlinear convergence (~1.6) using one function eval per iteration.
    """
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    # Ensure |fa| < |fb| initially? No, Illinois handles it.
    # Standard Illinois:
    # 1. Compute c = (a*fb - b*fa) / (fb - fa)
    # 2. If f(c) * f(a) < 0: b = c, fb = f(c)
    #    Else: a = c, fa = f(c), and fa = fa / 2.0 (shrink the weight of the older point)
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 1.0:
            break
        
        if abs(fa) <= 1e-8:
            return a, fa, True
        if abs(fb) <= 1e-8:
            return b, fb, True
        
        if fa == fb:
            break
            
        c = (a * fb - b * fa) / (fb - fa)
        
        if not math.isfinite(c):
            break
            
        fc = _safe_call(f, c)
        counter[0] += 1
        
        if math.isnan(fc):
            break
            
        if abs(fc) <= 1e-8:
            return c, fc, True
            
        if fa * fc < 0:
            # Root is in [a, c]
            b = c
            fb = fc
        else:
            # Root is in [c, b]
            a = c
            fa = fc
            # Illinois modification: shrink fa to accelerate convergence
            fa *= 0.5
            
        if abs(b - a) < 1e-14 * max(1.0, abs(c)):
            break
            
    # Return the best point found
    if abs(fa) < abs(fb):
        return a, fa, abs(fa) <= 1e-8
    else:
        return b, fb, abs(fb) <= 1e-8

def _find_bracket(f, x0, counter):
    """
    Find a bracket [a, b] such that f(a)*f(b) < 0.
    Uses exponential expansion from x0.
    """
    if counter[0] >= _CALL_BUDGET or _time_left() < 1.0:
        return None

    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return (x0, x0, fx0, fx0)

    # Try expanding in both directions
    # Direction +1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 5 or _time_left() < 1.0:
            break
        b = a + step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    # Direction -1
    a = x0
    fa = fx0
    step = 1e-3
    for _ in range(50):
        if counter[0] >= _CALL_BUDGET - 5 or _time_left() < 1.0:
            break
        b = a - step
        fb = _safe_call(f, b)
        counter[0] += 1
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return (a, b, fa, fb)
        a = b
        fa = fb
        step *= 2.0
        if step > 1e6:
            break

    return None

def _brent_like(f, a, b, fa, fb, counter):
    """
    Use scipy's brentq if available, otherwise fall back to Illinois/Bisection.
    """
    if _HAS_SCIPY:
        def wrapped_f(x):
            if counter[0] >= _CALL_BUDGET or _time_left() < 1.0:
                raise Exception("budget")
            val = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(val):
                raise Exception("nan")
            return val
        
        try:
            x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
            fx = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(fx):
                return x, fx, False
            return x, fx, abs(fx) <= 1e-8
        except Exception:
            pass

    # Fallback: Illinois method
    x, fx, success = _illinois_method(f, a, b, fa, fb, counter)
    return x, fx, success

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Try Secant first (fast local convergence)
    x, fx, success = _secant_method(f, x0, counter)
    if success:
        return x

    best_x = x
    best_fx = fx
    best_abs = abs(fx) if not math.isnan(fx) else float('inf')

    # 2. If Secant failed, try to find a bracket
    if counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        bracket = _find_bracket(f, x0, counter)
        if bracket:
            a, b, fa, fb = bracket
            if a == b:
                return a
            x, fx, success = _brent_like(f, a, b, fa, fb, counter)
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x = x
                    best_fx = fx
                    best_abs = abs(fx)
                if success:
                    return x

    # 3. If still no success, try Secant from the best point found so far
    #    (might be different if bracketing moved us closer)
    if counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        x, fx, success = _secant_method(f, best_x, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    if math.isnan(best_x):
        return x0

    return best_x
