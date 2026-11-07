"""
Root finder for 3 fixed oscillatory equations using IQI (Inverse Quadratic Interpolation)
as the primary solver with robust bracketing.

Strategy:
  1. Evaluate f at x0. If already good enough, return immediately.
  2. Find a bracket [a, b] with sign change using geometric expansion.
  3. Use IQI (inverse quadratic interpolation) for fast convergence (1 call/step).
     Falls back to bisection when IQI step is outside the bracket.
  4. If no bracket found, use secant method.
  5. Final polish with a few secant/IQI steps.

Call budget: keep under 480 calls per equation.
Time deadline: return best result before 140 s.
"""
import math
import time

_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480
_TOL = 1e-8


def _time_left():
    return _DEADLINE - time.time()


class _Counter:
    __slots__ = ('n',)
    def __init__(self):
        self.n = 0
    def __add__(self, other):
        self.n += other
        return self
    def __radd__(self, other):
        self.n += other
        return self


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


def _iqi_solve(f, a, b, fa, fb, ctr, max_iter=80):
    """
    Inverse Quadratic Interpolation solver.
    Requires a bracket [a, b] with fa * fb <= 0.
    Maintains three points and uses IQI to estimate the next point.
    Falls back to bisection when IQI step is unreliable.
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

    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    # Third point: start with midpoint for IQI initialization
    c = 0.5 * (a + b)
    fc = _safe_call(f, c, ctr)
    if math.isnan(fc):
        # Fallback to bisection
        return _bisection(f, a, b, fa, fb, ctr, max_iter)

    # Track the three most recent points: (x, fx)
    # We'll maintain x0, x1, x2 with corresponding f0, f1, f2
    x0, f0 = a, fa
    x1, f1 = c, fc
    x2, f2 = b, fb

    best_x = x0 if abs(f0) < abs(f2) else x2
    best_f = f0 if abs(f0) < abs(f2) else f2

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if abs(best_f) <= _TOL:
            return best_x, best_f, True

        # Update bracket: keep the two points that bracket the root
        # The root is between the two points with opposite signs
        # Find which pair brackets
        if f0 * f1 <= 0:
            a, fa = x0, f0
            b, fb = x1, f1
        elif f1 * f2 <= 0:
            a, fa = x1, f1
            b, fb = x2, f2
        else:
            # f0 and f2 should bracket (they had opposite signs initially)
            # If not, something went wrong; use bisection
            a, fa = x0, f0
            b, fb = x2, f2

        if fa * fb > 0:
            # No valid bracket, use bisection
            return _bisection(f, a, b, fa, fb, ctr, max_iter - 20)

        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(fb) <= _TOL:
            return b, fb, True

        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

        # IQI: find x such that f(x) = 0 using quadratic through (x0,f0), (x1,f1), (x2,f2)
        # Inverse: x = Q(f) where Q is quadratic in f
        # Using the formula from Numerical Recipes:
        # x = x0 * (x1-x2)*f0/((f0-f1)*(f0-f2)) + ... (Lagrange interpolation on inverse)
        # But we need to be careful about which points to use.
        # Standard IQI uses the three points, picking the one with smallest |f| as the "target"
        
        # Use the three points. The one with smallest |f| is where we want to converge.
        # Let's use the standard IQI formula.
        # We need f0 != f1, f0 != f2, f1 != f2 for the formula to work.
        
        # Use points (x0,f0), (x1,f1), (x2,f2)
        # IQI step:
        denom0 = (f0 - f1) * (f0 - f2)
        denom1 = (f1 - f0) * (f1 - f2)
        denom2 = (f2 - f0) * (f2 - f1)
        
        x_new = None
        if abs(denom0) > 1e-30 and abs(denom1) > 1e-30 and abs(denom2) > 1e-30:
            # Lagrange interpolation: x = sum(x_i * L_i(0)) where L_i are Lagrange polynomials in f
            # L_0(0) = (0 - f1)(0 - f2) / ((f0 - f1)(f0 - f2)) = f1*f2 / denom0
            # L_1(0) = f0*f2 / denom1
            # L_2(0) = f0*f1 / denom2
            x_new = x0 * (f1 * f2) / denom0 + x1 * (f0 * f2) / denom1 + x2 * (f0 * f1) / denom2
        
        if x_new is None or not math.isfinite(x_new):
            # Bisection fallback
            x_new = 0.5 * (a + b)
        
        # Check if x_new is within the bracket [min(a,b), max(a,b)]
        lo = min(a, b)
        hi = max(a, b)
        if x_new <= lo or x_new >= hi:
            x_new = 0.5 * (a + b)
        
        # Evaluate at x_new
        f_new = _safe_call(f, x_new, ctr)
        if math.isnan(f_new):
            break
        
        if abs(f_new) <= _TOL:
            return x_new, f_new, True
        
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
        
        # Update the three points: drop the one with largest |f|, add the new point
        # We want to keep the points that are most useful for IQI
        # Keep the two that bracket the root and the new point
        if f_new * fa <= 0:
            # x_new and a bracket the root; drop b (x2 or x1 depending)
            # Keep a, x_new, and the other bracketing point
            # Actually, let's keep the three points that give the best spread
            # Simple approach: keep a, x_new, and the point with smaller |f| among old non-bracketing
            # Let's just shift: x0=a, x1=x_new, x2= the other endpoint
            if f_new * fb <= 0:
                # Both bracket - pick the one closer to x_new for the third
                if abs(x_new - a) < abs(x_new - b):
                    x0, f0 = a, fa
                    x1, f1 = x_new, f_new
                    x2, f2 = b, fb
                else:
                    x0, f0 = b, fb
                    x1, f1 = x_new, f_new
                    x2, f2 = a, fa
            else:
                # x_new brackets with a, not b
                x0, f0 = a, fa
                x1, f1 = x_new, f_new
                x2, f2 = b, fb
        elif f_new * fb <= 0:
            # x_new brackets with b
            x0, f0 = b, fb
            x1, f1 = x_new, f_new
            x2, f2 = a, fa
        else:
            # x_new doesn't bracket with either - use bisection next
            # Keep the original bracket and the new point
            x0, f0 = a, fa
            x1, f1 = x_new, f_new
            x2, f2 = b, fb
            # Next iteration will detect no bracket and bisect

    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= _TOL
    return b, fb, abs(fb) <= _TOL


def _bisection(f, a, b, fa, fb, ctr, max_iter=60):
    """
    Bisection method. Requires fa * fb <= 0.
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
        mid = 0.5 * (a + b)
        fm = _safe_call(f, mid, ctr)
        if math.isnan(fm):
            break
        if abs(fm) <= _TOL:
            return mid, fm, True
        if fa * fm <= 0:
            b = mid
            fb = fm
        else:
            a = mid
            fa = fm
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

    if abs(fa) <= abs(fb):
        return a, fa, abs(fa) <= _TOL
    return b, fb, abs(fb) <= _TOL


def _secant(f, x0, fx0, ctr, max_iter=40):
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
        x0, fx0 = x1, fx1
        x1 = x_new
        fx1 = _safe_call(f, x1, ctr)
        if math.isnan(fx1):
            break
        if abs(fx1) <= _TOL:
            return x1, fx1, True
        if abs(fx0) <= _TOL:
            return x0, fx0, True

    if abs(fx0) <= abs(fx1):
        return x0, fx0, abs(fx0) <= _TOL
    return x1, fx1, abs(fx1) <= _TOL


def _find_bracket(f, x0, fx0, ctr, max_expand=25):
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

    return None


def _try_bracket_and_solve(f, x_start, fx_start, ctr):
    """Try to bracket around x_start and solve with IQI."""
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
        x, fx, _ = _iqi_solve(f, a, b, fa, fb, ctr)
        if not math.isnan(fx):
            return x, fx
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

    # Strategy 1: bracket + IQI
    result = _try_bracket_and_solve(f, x0, fx0, ctr)
    if result is not None:
        x, fx = result
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x, best_fx, best_abs = x, fx, abs(fx)
            if abs(fx) <= _TOL:
                # Quick polish with secant
                x_p, fx_p, _ = _secant(f, x, fx, ctr, max_iter=5)
                if not math.isnan(fx_p) and abs(fx_p) <= _TOL:
                    return x_p
                return best_x

    # Strategy 2: secant
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=30)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 3: try offsets
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
        result = _try_bracket_and_solve(f, x_try, fx_try, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 4: secant from best
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=20)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 5: try to bracket from best point
    if best_abs > _TOL and not math.isnan(best_fx):
        result = _try_bracket_and_solve(f, best_x, best_fx, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    return best_x
