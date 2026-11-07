import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 145.0
_CALL_BUDGET = 495

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

def _solve_bracket(f, a, b, fa, fb, counter):
    """Solve for root in bracket [a,b] using scipy or regula falsi fallback."""
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
            x_root = _scipy_brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50)
            fx_root = _safe_call(f, x_root)
            counter[0] += 1
            if not math.isnan(fx_root) and abs(fx_root) <= 1e-8:
                return x_root
        except Exception:
            pass
    
    # Fallback: Regula Falsi (false position) - faster than bisection for smooth functions
    a, b = float(a), float(b)
    fa, fb = float(fa), float(fb)
    for _ in range(80):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        # Regula falsi step
        denom = fb - fa
        if abs(denom) < 1e-300:
            # Fall back to bisection if nearly parallel
            c = (a + b) / 2.0
        else:
            c = (a * fb - b * fa) / denom
            # Ensure c is within [a, b]
            if c <= a or c >= b:
                c = (a + b) / 2.0
        fc = _safe_call(f, c)
        counter[0] += 1
        if math.isnan(fc):
            break
        if abs(fc) <= 1e-8:
            return c
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
        # If interval is too small, stop
        if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
            break
    return (a + b) / 2.0

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
    if not math.isnan(fx0) and abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    if math.isnan(fx0):
        # If f(x0) is NaN, try nearby points
        for h in [1e-4, -1e-4, 1e-3, -1e-3, 1e-2, -1e-2]:
            x_try = x0 + h
            fx_try = _safe_call(f, x_try)
            counter[0] += 1
            if not math.isnan(fx_try):
                if abs(fx_try) < best_abs:
                    best_x = x_try
                    best_fx = fx_try
                    best_abs = abs(fx_try)
                if abs(fx_try) <= 1e-8:
                    return x_try
        return best_x if not math.isnan(best_x) else x0

    # 2. Try Newton step to get a better point or bracket
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    
    points = [(x0, fx0)]
    if not math.isnan(fx1):
        if abs(fx1) < best_abs:
            best_x = x1
            best_fx = fx1
            best_abs = abs(fx1)
        if abs(fx1) <= 1e-8:
            return x1
        points.append((x1, fx1))
        
        # Newton step
        if fx1 != fx0:
            slope = (fx1 - fx0) / (x1 - x0)
            if slope != 0:
                x_new = x0 - fx0 / slope
                if math.isfinite(x_new):
                    fx_new = _safe_call(f, x_new)
                    counter[0] += 1
                    if not math.isnan(fx_new):
                        if abs(fx_new) < best_abs:
                            best_x = x_new
                            best_fx = fx_new
                            best_abs = abs(fx_new)
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        points.append((x_new, fx_new))
    
    # Check all pairs for a bracket
    for i in range(len(points)):
        for j in range(i+1, len(points)):
            xa, fa = points[i]
            xb, fb = points[j]
            if fa * fb < 0:
                result = _solve_bracket(f, xa, xb, fa, fb, counter)
                if math.isfinite(result):
                    return result
    
    # 3. Alternating exponential bracketing - search both directions simultaneously
    a_pos = x0
    fa_pos = fx0
    a_neg = x0
    fa_neg = fx0
    step = 1e-3
    
    for _ in range(60):
        if counter[0] >= _CALL_BUDGET - 10 or _time_left() < 1.0:
            break
        
        # Try + direction
        b_pos = a_pos + step
        fb_pos = _safe_call(f, b_pos)
        counter[0] += 1
        if not math.isnan(fb_pos):
            if abs(fb_pos) < best_abs:
                best_x = b_pos
                best_fx = fb_pos
                best_abs = abs(fb_pos)
            if abs(fb_pos) <= 1e-8:
                return b_pos
            if fb_pos * fa_pos < 0:
                result = _solve_bracket(f, a_pos, b_pos, fa_pos, fb_pos, counter)
                if math.isfinite(result):
                    return result
            a_pos = b_pos
            fa_pos = fb_pos
        else:
            # NaN in + direction, stop expanding +
            pass
        
        # Try - direction
        b_neg = a_neg - step
        fb_neg = _safe_call(f, b_neg)
        counter[0] += 1
        if not math.isnan(fb_neg):
            if abs(fb_neg) < best_abs:
                best_x = b_neg
                best_fx = fb_neg
                best_abs = abs(fb_neg)
            if abs(fb_neg) <= 1e-8:
                return b_neg
            if fb_neg * fa_neg < 0:
                result = _solve_bracket(f, a_neg, b_neg, fa_neg, fb_neg, counter)
                if math.isfinite(result):
                    return result
            a_neg = b_neg
            fa_neg = fb_neg
        else:
            # NaN in - direction, stop expanding -
            pass
        
        step *= 2.0
        if step > 1e6:
            break
    
    # If no bracket found, return best point
    return best_x
