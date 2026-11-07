"""Rewritten program with a robust multi-method root finder and time budget guard."""
import math
import time

# Global deadline: evaluator kills after 150s, we use 140s as a safety margin
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480  # Stay well under 500 per equation


def _time_left():
    return _DEADLINE - time.time()


def _safe_call(f, x, counter):
    """Call f(x) safely, returning nan on failure. Increments counter."""
    try:
        val = f(x)
        counter[0] += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        counter[0] += 1
        return float('nan')


def _secant(f, x0, counter, max_iter=40):
    """
    Secant method. Returns (x, fx, success).
    Uses 2 initial points then 1 call per iteration.
    """
    x = float(x0)
    # We need two points. Start with x0 and x0 + small offset.
    h = 1e-6 * max(1.0, abs(x))
    x_prev = x
    x_curr = x + h
    
    fx_prev = _safe_call(f, x_prev, counter)
    fx_curr = _safe_call(f, x_curr, counter)
    
    if math.isnan(fx_curr) and not math.isnan(fx_prev):
        x_curr, x_prev = x_prev, x_curr
        fx_curr, fx_prev = fx_prev, fx_curr
    if math.isnan(fx_curr):
        return x_curr, fx_curr, False
    
    if abs(fx_curr) <= 1e-8:
        return x_curr, fx_curr, True
    if abs(fx_prev) <= 1e-8:
        return x_prev, fx_prev, True

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fx_prev == fx_curr:
            break
        # Secant step
        x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        if not math.isfinite(x_new):
            break
        fx_new = _safe_call(f, x_new, counter)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new, fx_new, True
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_new, fx_new
    
    # Return the point with smaller |f|
    if abs(fx_prev) < abs(fx_curr):
        return x_prev, fx_prev, abs(fx_prev) <= 1e-8
    else:
        return x_curr, fx_curr, abs(fx_curr) <= 1e-8


def _bisection(f, a, b, fa, fb, counter, max_iter=80):
    """Bisection method. Returns (x, fx, success)."""
    if fa * fb > 0:
        # No bracket, return the better one
        if abs(fa) < abs(fb):
            return a, fa, abs(fa) <= 1e-8
        else:
            return b, fb, abs(fb) <= 1e-8
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # No bracket
        c = (a + b) / 2.0
        fc = _safe_call(f, c, counter)
        if math.isnan(fc):
            break
        if abs(fc) <= 1e-8:
            return c, fc, True
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) < 1e-12 * max(1.0, abs(c)):
            break
    # Return the midpoint of the final bracket
    c = (a + b) / 2.0
    fc = _safe_call(f, c, counter)
    return c, fc, abs(fc) <= 1e-8


def _brentq_like(f, a, b, fa, fb, counter, max_iter=60):
    """
    Brent's method (simplified) for finding a root in [a, b] where fa*fb < 0.
    Returns (x, fx, success).
    """
    # Ensure fa < 0 and fb > 0 for consistency
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    tol = 1e-12

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # Lost bracket
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(e) < tol:
            break  # Converged
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        # Attempt interpolation
        if abs(d) < tol:
            break
        # Secant step
        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = (fc - fa)
        if abs(p) >= abs(q):
            # Use secant
            s = p / q
        else:
            # Use quadratic interpolation
            s = m
            q = fc - fa
            r = fc - fb
            if abs(q - r) < 1e-30:
                s = m * (1.0 if p > 0 else -1.0)
            else:
                s = m * (p / (q - r))
        
        # Check if s is acceptable
        if ((3.0 * m - abs(e)) / 2.0 < s < abs(e) / 2.0) or \
           ((e == d and (s < 0 and fc < 0 or s > 0 and fc > 0)) or \
            (abs(s) >= abs(e) / 2.0)):
            s = m
            d = s
        else:
            e = d

        a = b
        fa = fb
        b = b + s
        fb = _safe_call(f, b, counter)
        if math.isnan(fb):
            break
        if abs(fb) <= 1e-8:
            return b, fb, True

    # Return best estimate
    return b, fb, abs(fb) <= 1e-8


def _find_bracket(f, x0, counter):
    """
    Try to find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Returns (a, b, fa, fb) or None.
    """
    fx0 = _safe_call(f, x0, counter)
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return x0, x0, fx0, fx0

    # Expand outward to find a sign change
    step = 1.0
    a = x0
    fa = fx0

    for _ in range(30):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        b = x0 + step
        fb = _safe_call(f, b, counter)
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return a, b, fa, fb
        # Also check the other direction
        b2 = x0 - step
        fb2 = _safe_call(f, b2, counter)
        if math.isnan(fb2):
            break
        if fa * fb2 < 0:
            return b2, a, fb2, fa
        step *= 2.0

    return None


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    best_x = x0
    best_fx = _safe_call(f, x0, counter)
    best_abs = abs(best_fx) if not math.isnan(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return x0

    # Strategy 1: Try Secant method first (fast, 1 call/iter after initial 2)
    x_sec, fx_sec, success = _secant(f, x0, counter, max_iter=30)
    if not math.isnan(fx_sec):
        if abs(fx_sec) < best_abs:
            best_x = x_sec
            best_abs = abs(fx_sec)
        if success:
            return x_sec

    # Strategy 2: Try to find a bracket and use Brent's method
    bracket = _find_bracket(f, x0, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if fa == fb and abs(fa) <= 1e-8:
            return a
        if fa * fb < 0:
            x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=50)
            if not math.isnan(fx_brent):
                if abs(fx_brent) < best_abs:
                    best_x = x_brent
                    best_abs = abs(fx_brent)
                if success:
                    return x_brent
            # Fallback to bisection if Brent didn't converge
            x_bis, fx_bis, success = _bisection(f, a, b, fa, fb, counter, max_iter=80)
            if not math.isnan(fx_bis):
                if abs(fx_bis) < best_abs:
                    best_x = x_bis
                    best_abs = abs(fx_bis)
                if success:
                    return x_bis

    # Strategy 3: Try more Secant iterations from the best point found
    if best_abs > 1e-8 and not math.isnan(best_fx):
        x_sec2, fx_sec2, success = _secant(f, best_x, counter, max_iter=25)
        if not math.isnan(fx_sec2):
            if abs(fx_sec2) < best_abs:
                best_x = x_sec2
                best_abs = abs(fx_sec2)
            if success:
                return x_sec2

    # Strategy 4: Try a few more bracketing attempts with different starting points
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        x_try = x0 + offset
        bracket = _find_bracket(f, x_try, counter)
        if bracket is not None:
            a, b, fa, fb = bracket
            if fa * fb < 0:
                x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=40)
                if not math.isnan(fx_brent):
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_abs = abs(fx_brent)
                    if success:
                        return x_brent

    # Return the best result found
    return best_x
