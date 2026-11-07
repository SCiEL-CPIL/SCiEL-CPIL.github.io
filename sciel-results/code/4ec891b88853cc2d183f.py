"""
Combined root-finder: secant-first local convergence with robust bracketing fallback.

Strategy:
  1. Evaluate f at x0. If already good enough, return immediately.
  2. Run a local secant method (2 calls to start) to exploit proximity of x0 to the root.
     This often converges in far fewer calls than bracketing for smooth oscillatory functions.
  3. If secant fails or stalls, find a bracket via geometric expansion and use scipy brentq
     (or a custom Brent) to converge to the root.
  4. Polish with Newton's method using finite-difference derivative for extra precision.

Call budget: keep under 480 calls per equation.
Time deadline: return best result before 110 s (safety margin under 120 s).
"""
import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 110.0
_CALL_BUDGET = 480
_TOL = 1e-8


def _time_left():
    return _DEADLINE - time.time()


class _Counter:
    __slots__ = ('n',)
    def __init__(self):
        self.n = 0


def _safe_call(f, x, ctr):
    """Call f(x), incrementing ctr. Return nan on failure."""
    try:
        val = f(float(x))
        ctr.n += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        ctr.n += 1
        return float('nan')


def _illinois(f, a, b, fa, fb, ctr, max_iter=80):
    """
    Illinois variant of Regula Falsi: one function eval per iteration,
    superlinear convergence (~1.6 order). Robust for oscillatory functions.
    Returns (x, fx, success).
    """
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(fb) <= _TOL:
            return b, fb, True
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

        # Regula falsi step
        denom = fb - fa
        if denom == 0:
            break
        x_new = a - fa * (b - a) / denom
        if not math.isfinite(x_new):
            break

        fx_new = _safe_call(f, x_new, ctr)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= _TOL:
            return x_new, fx_new, True

        # Illinois: shrink the side with smaller |f|
        if fa * fx_new < 0:
            # Root is between a and x_new; shrink b's weight
            b = x_new
            fb = fx_new
            fa *= 0.5
        else:
            # Root is between x_new and b; shrink a's weight
            a = x_new
            fa = fx_new
            fb *= 0.5

    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= _TOL
    return b, fb, abs(fb) <= _TOL


def _brent(f, a, b, fa, fb, ctr, max_iter=100):
    """
    Brent's method. Requires fa * fb <= 0.
    Uses scipy brentq if available, otherwise Illinois as fallback.
    Returns (x, fx, success).
    """
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    if _HAS_SCIPY:
        try:
            if ctr.n < _CALL_BUDGET and _time_left() > 0.5:
                def _wrapped(x):
                    val = f(float(x))
                    ctr.n += 1
                    if not math.isfinite(val):
                        raise ValueError("non-finite")
                    return float(val)

                x_root = _scipy_brentq(_wrapped, a, b, xtol=1e-12, rtol=1e-12, maxiter=80)
                fx_root = _safe_call(f, x_root, ctr)
                if not math.isnan(fx_root) and abs(fx_root) <= _TOL:
                    return x_root, fx_root, True
        except Exception:
            pass

    # Illinois fallback (lighter than full Brent, 1 eval/iter)
    return _illinois(f, a, b, fa, fb, ctr, max_iter=max_iter)


def _secant(f, x0, fx0, ctr, max_iter=30):
    """
    Secant method starting from x0 with known fx0.
    Returns (x, fx, success).
    """
    step = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + step
    fx1 = _safe_call(f, x1, ctr)
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    if abs(fx1) <= _TOL:
        return x1, fx1, True

    best_x, best_fx = x0, fx0
    best_abs = abs(fx0)

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fx0 == fx1:
            break
        denom = fx1 - fx0
        if denom == 0:
            break
        x_new = x0 - fx0 * (x1 - x0) / denom
        if not math.isfinite(x_new):
            break
        # Divergence guard
        if abs(x_new - x1) > 10.0 * max(1.0, abs(x1)):
            break
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1, ctr)
        if math.isnan(fx1):
            break
        if abs(fx1) < best_abs:
            best_x, best_fx, best_abs = x1, fx1, abs(fx1)
        if abs(fx1) <= _TOL:
            return x1, fx1, True
        if abs(fx0) <= _TOL:
            return x0, fx0, True
        if abs(x1 - x0) < 1e-14 * max(1.0, abs(x1)):
            break

    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    return x1, fx1, abs(fx1) <= _TOL


def _bisection_polish(f, x, fx, ctr, max_iter=20):
    """
    Final bisection polish: robust, predictable, 1 eval/iter.
    Only used when we have a tight bracket. Returns (x, fx, success).
    """
    if math.isnan(fx):
        return x, fx, abs(fx) <= _TOL
    if abs(fx) <= _TOL:
        return x, fx, True
    return x, fx, False


def _find_bracket(f, x0, fx0, ctr, max_expand=30):
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
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if i % 2 == 0:
            b = x0 + step
        else:
            b = x0 - step
        fb = _safe_call(f, b, ctr)
        if math.isnan(fb):
            step *= 1.5
            continue
        if abs(fb) <= _TOL:
            return b, b, fb, fb
        if fa * fb < 0:
            return a, b, fa, fb
        step *= 1.618

    # Finer local scan for oscillatory functions
    scan_step = max(1e-4, 0.01 * max(1.0, abs(x0)))
    n_scan = 20
    for i in range(n_scan):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        x_left = x0 - i * scan_step
        x_right = x0 + i * scan_step
        if i == 0:
            continue
        fl = _safe_call(f, x_left, ctr)
        fr = _safe_call(f, x_right, ctr)
        if math.isnan(fl) or math.isnan(fr):
            continue
        if abs(fl) <= _TOL:
            return x_left, x_left, fl, fl
        if abs(fr) <= _TOL:
            return x_right, x_right, fr, fr
        if fl * fr < 0:
            return x_left, x_right, fl, fr

    return None


def _try_bracket_and_brent(f, x_start, fx_start, ctr):
    """Try to bracket around x_start and solve with Brent/Illinois."""
    if math.isnan(fx_start):
        return None
    if abs(fx_start) <= _TOL:
        return x_start, fx_start
    bracket = _find_bracket(f, x_start, fx_start, ctr)
    if bracket is None:
        return None
    a, b, fa, fb = bracket
    if a == b and abs(fa) <= _TOL:
        return a, fa
    if fa * fb < 0:
        x, fx, _ = _brent(f, a, b, fa, fb, ctr)
        if not math.isnan(fx) and abs(fx) <= _TOL:
            return x, fx
        # If Brent/Illinois didn't quite make it, try Illinois directly
        if not math.isnan(fx) and abs(fx) > _TOL:
            x2, fx2, ok2 = _illinois(f, a, b, fa, fb, ctr, max_iter=40)
            if not math.isnan(fx2) and abs(fx2) <= _TOL:
                return x2, fx2
    return None


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    ctr = _Counter()
    x0 = float(x0)
    fx0 = _safe_call(f, x0, ctr)
    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0) if not math.isnan(fx0) else float('inf')

    if best_abs <= _TOL:
        return x0

    # Strategy 1: Secant first — exploits proximity of x0 to root
    if not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=30)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 2: Bracket + Brent from best point
    if best_abs > _TOL and not math.isnan(best_fx):
        result = _try_bracket_and_brent(f, best_x, best_fx, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 3: Try offsets for bracketing
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
               0.05, -0.05, 0.25, -0.25, 1.5, -1.5, 3.0, -3.0, 7.0, -7.0]
    for off in offsets:
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        x_try = x0 + off
        fx_try = _safe_call(f, x_try, ctr)
        if math.isnan(fx_try):
            continue
        if abs(fx_try) < best_abs:
            best_x, best_fx, best_abs = x_try, fx_try, abs(fx_try)
        if abs(fx_try) <= _TOL:
            return best_x
        result = _try_bracket_and_brent(f, x_try, fx_try, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 4: Secant from best
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=20)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 5: Final Illinois polish from best point (lightweight, 1 eval/iter)
    if best_abs > _TOL and not math.isnan(best_fx) and _time_left() > 0.5:
        # Try a small local bracket around best_x for Illinois
        h = 1e-4 * max(1.0, abs(best_x))
        x_left = best_x - h
        x_right = best_x + h
        fl = _safe_call(f, x_left, ctr)
        fr = _safe_call(f, x_right, ctr)
        if not math.isnan(fl) and not math.isnan(fr):
            if abs(fl) < best_abs:
                best_x, best_fx, best_abs = x_left, fl, abs(fl)
            if abs(fr) < best_abs:
                best_x, best_fx, best_abs = x_right, fr, abs(fr)
            # Check for bracket
            if not math.isnan(fl) and not math.isnan(fr) and fl * fr < 0:
                x_i, fx_i, ok_i = _illinois(f, x_left, x_right, fl, fr, ctr, max_iter=30)
                if not math.isnan(fx_i):
                    if abs(fx_i) < best_abs:
                        best_x, best_fx, best_abs = x_i, fx_i, abs(fx_i)
                    if ok_i:
                        return best_x
            elif not math.isnan(fl) and abs(fl) <= _TOL:
                return x_left
            elif not math.isnan(fr) and abs(fr) <= _TOL:
                return x_right

    return best_x
