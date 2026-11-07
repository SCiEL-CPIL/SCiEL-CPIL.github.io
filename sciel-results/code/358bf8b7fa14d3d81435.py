"""Combined root finder: IQI/Secant primary, robust bracketing, tight call budget."""
import math
import time

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 460
_TOL = 1e-8


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


def _iqi_secant(f, x0, fx0, counter, max_iter=30):
    """
    Inverse quadratic interpolation / secant hybrid.
    Maintains 3 points. Uses IQI when possible, falls back to secant.
    1 call per iteration after initial setup.
    Returns (x, fx, success).
    """
    # Initial points
    h = 0.1 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = _safe_call(f, x1, counter)
    if math.isnan(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    if abs(fx1) <= _TOL:
        return x1, fx1, True

    # Third point for IQI
    x2 = x0 - h
    fx2 = _safe_call(f, x2, counter)
    if math.isnan(fx2):
        # Fall back to secant with x0, x1
        return _secant_only(f, x0, fx0, x1, fx1, counter, max_iter)
    if abs(fx2) <= _TOL:
        return x2, fx2, True

    # Maintain points: (xa, fxa), (xb, fxb), (xc, fxc)
    # Sort by |f| so that xa has smallest |f|
    pts = [(x0, fx0), (x1, fx1), (x2, fx2)]
    pts.sort(key=lambda p: abs(p[1]))
    xa, fxa = pts[0]
    xb, fxb = pts[1]
    xc, fxc = pts[2]

    best_x, best_fx = xa, fxa

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break

        # Update best
        if abs(fxa) < abs(best_fx):
            best_x, best_fx = xa, fxa
        if abs(fxb) < abs(best_fx):
            best_x, best_fx = xb, fxb
        if abs(fxc) < abs(best_fx):
            best_x, best_fx = xc, fxc

        if abs(best_fx) <= _TOL:
            return best_x, best_fx, True

        # Check for duplicate function values
        if fxa == fxb or fxb == fxc or fxa == fxc:
            # Fall back to secant using the two distinct points
            if fxa != fxc:
                x_new = _secant_step(xa, fxa, xc, fxc)
            elif fxa != fxb:
                x_new = _secant_step(xa, fxa, xb, fxb)
            else:
                x_new = _secant_step(xb, fxb, xc, fxc)
        else:
            # Try IQI
            x_new = _iqi_step(xa, fxa, xb, fxb, xc, fxc)
            if x_new is None or not math.isfinite(x_new):
                # Fall back to secant on the two best points
                x_new = _secant_step(xa, fxa, xb, fxb)

        if not math.isfinite(x_new):
            break

        # Check if x_new is too close to existing points
        if min(abs(x_new - xa), abs(x_new - xb), abs(x_new - xc)) < 1e-14 * max(1.0, abs(x_new)):
            # Perturb slightly
            x_new = x_new + 1e-8 * max(1.0, abs(x_new))

        fx_new = _safe_call(f, x_new, counter)
        if math.isnan(fx_new):
            break

        if abs(fx_new) <= _TOL:
            return x_new, fx_new, True

        # Replace the worst point
        if abs(fx_new) < abs(fxc):
            xc, fxc = x_new, fx_new
        elif abs(fx_new) < abs(fxb):
            xb, fxb = x_new, fx_new
        else:
            xa, fxa = x_new, fx_new

        # Re-sort
        pts = [(xa, fxa), (xb, fxb), (xc, fxc)]
        pts.sort(key=lambda p: abs(p[1]))
        xa, fxa = pts[0]
        xb, fxb = pts[1]
        xc, fxc = pts[2]

    return best_x, best_fx, abs(best_fx) <= _TOL


def _secant_step(xa, fxa, xb, fxb):
    """Compute secant step from two points."""
    if fxa == fxb:
        return None
    return xa - fxa * (xa - xb) / (fxa - fxb)


def _iqi_step(xa, fxa, xb, fxb, xc, fxc):
    """
    Inverse quadratic interpolation step.
    Returns the interpolated x, or None if not applicable.
    """
    # Check for duplicates
    if fxa == fxb or fxb == fxc or fxa == fxc:
        return None

    # IQI formula
    # We want x such that f(x) = 0, interpolating 1/f
    # Using Lagrange interpolation on (x, 1/f)
    d1 = (xa - xb) * (xa - xc)
    d2 = (xb - xa) * (xb - xc)
    d3 = (xc - xa) * (xc - xb)

    if abs(d1) < 1e-30 or abs(d2) < 1e-30 or abs(d3) < 1e-30:
        return None

    term1 = d1 / fxa
    term2 = d2 / fxb
    term3 = d3 / fxc

    denom = term1 + term2 + term3
    if abs(denom) < 1e-30:
        return None

    x_new = (term1 * xa + term2 * xb + term3 * xc) / denom

    # Check if x_new is within a reasonable range
    x_min = min(xa, xb, xc)
    x_max = max(xa, xb, xc)
    margin = 0.5 * (x_max - x_min) if x_max > x_min else 1.0
    if x_new < x_min - margin or x_new > x_max + margin:
        return None

    return x_new


def _secant_only(f, xa, fxa, xb, fxb, counter, max_iter=25):
    """Simple secant method with two starting points."""
    best_x, best_fx = xa, fxa
    if abs(fxb) < abs(best_fx):
        best_x, best_fx = xb, fxb

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fxa == fxb:
            break
        x_new = _secant_step(xa, fxa, xb, fxb)
        if x_new is None or not math.isfinite(x_new):
            break
        fx_new = _safe_call(f, x_new, counter)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= _TOL:
            return x_new, fx_new, True
        if abs(fx_new) < abs(best_fx):
            best_x, best_fx = x_new, fx_new
        xa, fxa = xb, fxb
        xb, fxb = x_new, fx_new

    return best_x, best_fx, abs(best_fx) <= _TOL


def _bisection(f, a, b, fa, fb, counter, max_iter=80):
    """Bisection method. Returns (x, fx, success)."""
    best_x, best_fx = a, fa
    if abs(fb) < abs(best_fx):
        best_x, best_fx = b, fb

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
        c = (a + b) / 2.0
        fc = _safe_call(f, c, counter)
        if math.isnan(fc):
            break
        if abs(fc) <= _TOL:
            return c, fc, True
        if abs(fc) < abs(best_fx):
            best_x, best_fx = c, fc
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) < 1e-13 * max(1.0, abs(c)):
            break

    return best_x, best_fx, abs(best_fx) <= _TOL


def _brentq_like(f, a, b, fa, fb, counter, max_iter=50):
    """
    Brent's method (simplified) for finding a root in [a, b] where fa*fb < 0.
    Returns (x, fx, success).
    """
    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    best_x, best_fx = a, fa
    if abs(fb) < abs(best_fx):
        best_x, best_fx = b, fb

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(fb) <= _TOL:
            return b, fb, True
        if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
            break

        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(e) < 1e-13:
            break
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        # Inverse quadratic interpolation
        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = (fc - fa)
        if abs(p) >= abs(q):
            s = p / q if q != 0 else m
        else:
            s = m
            q = fc - fa
            r = fc - fb
            if abs(q - r) < 1e-30:
                s = m
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
        if abs(fb) <= _TOL:
            return b, fb, True
        if abs(fb) < abs(best_fx):
            best_x, best_fx = b, fb

    return best_x, best_fx, abs(best_fx) <= _TOL


def _find_bracket(f, x0, fx0, counter):
    """
    Find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Uses geometric expansion, alternating sides to save calls.
    Returns (a, b, fa, fb) or None.
    """
    if math.isnan(fx0):
        return None
    if abs(fx0) <= _TOL:
        return x0, x0, fx0, fx0

    step = max(1e-3, 0.1 * max(1.0, abs(x0)))
    a = x0
    fa = fx0

    for i in range(20):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if i % 2 == 0:
            b = x0 + step
            fb = _safe_call(f, b, counter)
            if not math.isnan(fb):
                if abs(fb) <= _TOL:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return a, b, fa, fb
        else:
            b = x0 - step
            fb = _safe_call(f, b, counter)
            if not math.isnan(fb):
                if abs(fb) <= _TOL:
                    return b, b, fb, fb
                if fa * fb < 0:
                    return b, a, fb, fa
        step *= 2.0
        if step > 1e8:
            break

    return None


def _polish(f, x, fx, counter, max_iter=5):
    """Final high-precision polish using secant with small step."""
    if math.isnan(fx):
        return x, fx, False
    if abs(fx) <= _TOL:
        return x, fx, True

    h = 1e-6 * max(1.0, abs(x))
    x_prev = x - h
    fx_prev = _safe_call(f, x_prev, counter)
    if math.isnan(fx_prev):
        return x, fx, abs(fx) <= _TOL

    best_x, best_fx = x, fx
    if abs(fx_prev) < abs(best_fx):
        best_x, best_fx = x_prev, fx_prev

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.1:
            break
        if fx == fx_prev:
            break
        x_new = _secant_step(x, fx, x_prev, fx_prev)
        if x_new is None or not math.isfinite(x_new):
            break
        fx_new = _safe_call(f, x_new, counter)
        if math.isnan(fx_new):
            break
        if abs(fx_new) <= _TOL:
            return x_new, fx_new, True
        if abs(fx_new) < abs(best_fx):
            best_x, best_fx = x_new, fx_new
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new

    return best_x, best_fx, abs(best_fx) <= _TOL


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

    if best_abs <= _TOL:
        return x0

    # Strategy 1: IQI/Secant hybrid (fast convergence, 1 call/iter after setup)
    x_iqi, fx_iqi, success = _iqi_secant(f, x0, best_fx, counter, max_iter=30)
    if not math.isnan(fx_iqi):
        if abs(fx_iqi) < best_abs:
            best_x = x_iqi
            best_abs = abs(fx_iqi)
        if success:
            # Polish for extra precision
            x_pol, fx_pol, _ = _polish(f, x_iqi, fx_iqi, counter, max_iter=4)
            if not math.isnan(fx_pol) and abs(fx_pol) < best_abs:
                best_x = x_pol
                best_abs = abs(fx_pol)
            return best_x

    # Strategy 2: Find bracket, then use Brent's method
    bracket = _find_bracket(f, x0, best_fx, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if a == b:
            return a
        if fa * fb < 0:
            x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=50)
            if not math.isnan(fx_brent):
                if abs(fx_brent) < best_abs:
                    best_x = x_brent
                    best_abs = abs(fx_brent)
                if success:
                    x_pol, fx_pol, _ = _polish(f, x_brent, fx_brent, counter, max_iter=3)
                    if not math.isnan(fx_pol) and abs(fx_pol) < best_abs:
                        best_x = x_pol
                        best_abs = abs(fx_pol)
                    return best_x
            # Fallback to bisection if Brent didn't converge
            x_bis, fx_bis, success = _bisection(f, a, b, fa, fb, counter, max_iter=60)
            if not math.isnan(fx_bis):
                if abs(fx_bis) < best_abs:
                    best_x = x_bis
                    best_abs = abs(fx_bis)
                if success:
                    return best_x

    # Strategy 3: Try IQI from the best point found
    if best_abs > _TOL and not math.isnan(best_fx):
        x_iqi2, fx_iqi2, success = _iqi_secant(f, best_x, best_fx, counter, max_iter=20)
        if not math.isnan(fx_iqi2):
            if abs(fx_iqi2) < best_abs:
                best_x = x_iqi2
                best_abs = abs(fx_iqi2)
            if success:
                return best_x

    # Strategy 4: Try a few more bracketing attempts with different starting points
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        x_try = x0 + offset
        bracket = _find_bracket(f, x_try, counter)
        if bracket is not None:
            a, b, fa, fb = bracket
            if a != b and fa * fb < 0:
                x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=40)
                if not math.isnan(fx_brent):
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_abs = abs(fx_brent)
                    if success:
                        return best_x

    # Return the best result found
    return best_x
