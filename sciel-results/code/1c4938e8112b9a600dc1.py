import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0
_CALL_BUDGET = 490
_TOL = 1e-8

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

def _secant(f, x0, counter, max_iter=50):
    """
    Secant method with safeguards.
    """
    x1 = x0
    # Choose a small step based on the magnitude of x0
    step = 1e-5 * max(1.0, abs(x0))
    x2 = x0 + step
    
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    if math.isnan(fx1):
        return x1, fx1, False
    
    fx2 = _safe_call(f, x2)
    counter[0] += 1
    if math.isnan(fx2):
        return x2, fx2, False
    
    if abs(fx1) <= _TOL:
        return x1, fx1, True
    if abs(fx2) <= _TOL:
        return x2, fx2, True

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
            
        if abs(fx2) <= _TOL:
            return x2, fx2, True
        if abs(fx1) <= _TOL:
            return x1, fx1, True
        
        if fx2 == fx1:
            break
            
        # Secant step
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        if not math.isfinite(x_new):
            break
            
        # Check for stagnation
        if abs(x_new - x2) < 1e-15 * max(1.0, abs(x2)):
            break

        x1, fx1 = x2, fx2
        x2 = x_new
        fx2 = _safe_call(f, x2)
        counter[0] += 1
        if math.isnan(fx2):
            break
            
    if abs(fx2) <= _TOL:
        return x2, fx2, True
    if abs(fx1) <= _TOL:
        return x1, fx1, True
    return x2, fx2, False

def _brent_like(f, a, b, fa, fb, counter):
    """
    Use scipy's brentq if available, otherwise fall back to Illinois/bisection.
    """
    if _HAS_SCIPY:
        def wrapped_f(x):
            if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                raise Exception("budget")
            val = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(val):
                raise Exception("nan")
            return val
        
        try:
            x = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
            fx = _safe_call(f, x)
            counter[0] += 1
            if math.isnan(fx):
                return x, fx, False
            return x, fx, abs(fx) <= _TOL
        except Exception:
            pass

    # Fallback: Illinois method
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(100):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
            
        # Illinois method
        if abs(fa) < abs(fb):
            c = (a + b) / 2.0
        else:
            c = (a * fb - b * fa) / (fb - fa)
            if not math.isfinite(c) or c <= a or c >= b:
                c = (a + b) / 2.0
                
        fc = _safe_call(f, c)
        counter[0] += 1
        if math.isnan(fc):
            break
        if abs(fc) <= _TOL:
            return c, fc, True
        if fa * fc < 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
            
        if abs(b - a) < 1e-14 * max(1.0, abs(c)):
            break
            
    c = (a + b) / 2.0
    fc = _safe_call(f, c)
    counter[0] += 1
    return c, fc, abs(fc) <= _TOL

def _find_bracket(f, x0, counter):
    """
    Find a bracket [a, b] such that f(a)*f(b) < 0.
    Uses exponential expansion from x0 in both directions.
    """
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return None
    if abs(fx0) <= _TOL:
        return (x0, x0, fx0, fx0)

    best = (x0, fx0, abs(fx0))
    
    # Try expanding in both directions
    for direction in (1, -1):
        a = x0
        fa = fx0
        step = 1e-3
        for _ in range(50):
            if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
                break
            b = a + direction * step
            fb = _safe_call(f, b)
            counter[0] += 1
            if math.isnan(fb):
                break
            if fa * fb < 0:
                return (a, b, fa, fb)
            if abs(fb) < best[2]:
                best = (b, fb, abs(fb))
            a = b
            fa = fb
            step *= 2.0
            if step > 1e6:
                break

    return None

def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    
    # 1. Check x0
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if not math.isnan(fx0) and abs(fx0) <= _TOL:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    # 2. Try local secant method first
    if counter[0] < _CALL_BUDGET - 20 and _time_left() > 1.0:
        x, fx, success = _secant(f, x0, counter)
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x = x
                best_fx = fx
                best_abs = abs(fx)
            if success:
                return x

    # 3. Try to find a bracket using exponential expansion
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

    # 4. If no bracket found or convergence failed, try secant from best point again
    if counter[0] < _CALL_BUDGET - 10 and _time_left() > 1.0:
        x, fx, success = _secant(f, best_x, counter)
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
