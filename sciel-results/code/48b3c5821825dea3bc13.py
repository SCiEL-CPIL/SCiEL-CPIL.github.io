"""
Root finder for 3 fixed oscillatory equations using a hybrid approach:
1. Try to find a bracket via geometric expansion.
2. Use Brent's method (via scipy) for robust, fast convergence within the bracket.
3. Fall back to secant method if no bracket is found.
4. Try multiple starting offsets to handle oscillatory functions with multiple roots.

The key insight is that scipy's brentq is highly optimized and uses
inverse quadratic interpolation internally, which is optimal for smooth
functions. By combining it with robust bracketing, we get both speed and
reliability.
"""
import math
import time

_DEADLINE = time.time() + 120.0
_TOL = 1e-8


def _time_left():
    return _DEADLINE - time.time()


def _safe_call(f, x, counter):
    """Call f(x), incrementing counter. Return nan on failure."""
    try:
        val = f(float(x))
        counter[0] += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        counter[0] += 1
        return float('nan')


def _find_bracket(f, x0, fx0, counter, max_expand=30):
    """
    Find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Expands geometrically, alternating direction.
    Returns (a, b, fa, fb) or None.
    """
    if math.isnan(fx0):
        return None
    if abs(fx0) <= _TOL:
        return x0, x0, fx0, fx0

    a = x0
    fa = fx0
    step = max(1e-3, 0.05 * max(1.0, abs(x0)))

    for i in range(max_expand):
        if counter[0] >= 450 or _time_left() < 1.0:
            break
        if i % 2 == 0:
            b = x0 + step
        else:
            b = x0 - step
        fb = _safe_call(f, b, counter)
        if math.isnan(fb):
            step *= 1.5
            continue
        if abs(fb) <= _TOL:
            return b, b, fb, fb
        if fa * fb < 0:
            return a, b, fa, fb
        step *= 1.618

    return None


def _brent_solve(f, a, b, fa, fb, counter):
    """
    Use scipy's brentq to find root in [a, b].
    We wrap f to count calls.
    Returns (x, fx, success) or None on failure.
    """
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        return None

    from scipy.optimize import brentq

    # Wrap f to count calls
    def wrapped_f(x):
        val = _safe_call(f, x, counter)
        if math.isnan(val):
            # Return a large value to signal failure; brentq may handle this
            return float('inf') if x > 0 else float('-inf')
        return val

    try:
        x_root = brentq(wrapped_f, a, b, xtol=1e-14, rtol=1e-14, maxiter=100)
        fx_root = _safe_call(f, x_root, counter)
        if math.isnan(fx_root):
            return None
        return x_root, fx_root, abs(fx_root) <= _TOL
    except Exception:
        return None


def _secant_solve(f, x0, fx0, counter, max_iter=50):
    """
    Secant method starting from x0 with known fx0.
    Returns (x, fx, success).
    """
    step = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = _safe_call(f, x1, counter)
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    if abs(fx1) <= _TOL:
        return x1, fx1, True

    best_x = x0 if abs(fx0) < abs(fx1) else x1
    best_fx = fx0 if abs(fx0) < abs(fx1) else fx1

    for _ in range(max_iter):
        if counter[0] >= 450 or _time_left() < 1.0:
            break
        if abs(best_fx) <= _TOL:
            return best_x, best_fx, True
        if fx0 == fx1:
            break
        denom = fx1 - fx0
        if abs(denom) < 1e-30:
            break
        x_new = x0 - fx0 * (x1 - x0) / denom
        if not math.isfinite(x_new):
            break
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1, counter)
        if math.isnan(fx1):
            break
        if abs(fx1) < abs(best_fx):
            best_x, best_fx = x1, fx1
        if abs(fx0) < abs(best_fx):
            best_x, best_fx = x0, fx0

    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    return x1, fx1, abs(fx1) <= _TOL


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    fx0 = _safe_call(f, x0, counter)
    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    if best_abs <= _TOL:
        return x0

    # Strategy 1: bracket + Brent's method
    bracket = _find_bracket(f, x0, fx0, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b and abs(fa) <= _TOL:
            return a
        if fa * fb < 0:
            result = _brent_solve(f, a, b, fa, fb, counter)
            if result is not None:
                x, fx, ok = result
                if not math.isnan(fx):
                    if abs(fx) < best_abs:
                        best_x, best_fx, best_abs = x, fx, abs(fx)
                    if ok:
                        return best_x

    # Strategy 2: secant method
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant_solve(f, best_x, best_fx, counter, max_iter=30)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 3: try multiple offsets to find a bracket
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
               0.05, -0.05, 0.25, -0.25, 1.5, -1.5, 3.0, -3.0, 7.0, -7.0,
               10.0, -10.0, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
    for off in offsets:
        if counter[0] >= 450 or _time_left() < 1.0:
            break
        x_try = x0 + off
        fx_try = _safe_call(f, x_try, counter)
        if math.isnan(fx_try):
            continue
        if abs(fx_try) < best_abs:
            best_x, best_fx, best_abs = x_try, fx_try, abs(fx_try)
        if abs(fx_try) <= _TOL:
            return best_x

        # Try to bracket from this point
        bracket = _find_bracket(f, x_try, fx_try, counter, max_expand=15)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a == b and abs(fa) <= _TOL:
                return a
            if fa * fb < 0:
                result = _brent_solve(f, a, b, fa, fb, counter)
                if result is not None:
                    x, fx, ok = result
                    if not math.isnan(fx):
                        if abs(fx) < best_abs:
                            best_x, best_fx, best_abs = x, fx, abs(fx)
                        if ok:
                            return best_x

    # Strategy 4: final secant attempt
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant_solve(f, best_x, best_fx, counter, max_iter=20)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    return best_x
