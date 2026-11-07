"""
Direct approach for root-finding on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If already good enough, return immediately.
  2. Build a local Taylor polynomial (degree 4) using 5-point finite differences.
     Use the polynomial to predict root candidates via Newton iterations on the
     local model (no additional f calls). Verify candidates with f.
  3. If Taylor-based candidates don't converge, find a bracket [a, b] with
     sign change using geometric expansion and use scipy brentq (or custom Brent).
  4. Fall back to secant method if bracketing fails.
  5. Polish with Newton's method using finite-difference derivative.

Call budget: keep under 500 calls per equation.
Time deadline: return best result before 130 s.
"""
import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 130.0
_CALL_BUDGET = 450
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


def _brent(f, a, b, fa, fb, ctr, max_iter=100):
    """
    Brent's method. Requires fa * fb <= 0.
    Uses scipy brentq if available for speed, otherwise custom implementation.
    Returns (x, fx, success).
    """
    if abs(fa) <= _TOL:
        return a, fa, True
    if abs(fb) <= _TOL:
        return b, fb, True
    if fa * fb > 0:
        # No sign change; return the one with smaller |f|
        if abs(fa) <= abs(fb):
            return a, fa, abs(fa) <= _TOL
        return b, fb, abs(fb) <= _TOL

    # Try scipy brentq first — it's highly optimized
    if _HAS_SCIPY:
        try:
            if ctr.n >= _CALL_BUDGET or _time_left() < 0.5:
                pass
            else:
                # Wrap f to track calls
                def _wrapped(x):
                    val = f(float(x))
                    ctr.n += 1
                    if not math.isfinite(val):
                        raise ValueError("non-finite")
                    return float(val)
                
                x_root = _scipy_brentq(_wrapped, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                fx_root = _safe_call(f, x_root, ctr)
                if not math.isnan(fx_root) and abs(fx_root) <= _TOL:
                    return x_root, fx_root, True
                # Even if not below TOL, return if it's the best we have
                if not math.isnan(fx_root):
                    return x_root, fx_root, abs(fx_root) <= _TOL
        except Exception:
            pass  # Fall through to custom Brent

    # Custom Brent's method as fallback
    # Ensure |fa| <= |fb| so we move b
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if fa * fb > 0:
            break
        if abs(fa) <= _TOL:
            return a, fa, True
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

        # Inverse quadratic interpolation
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(e) < 1e-14:
            break
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = fc - fa
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

        # Check if s is acceptable, otherwise bisect
        if ((3.0 * m - abs(e)) / 2.0 < s < abs(e) / 2.0) or \
           ((e == d and (s < 0 and fc < 0 or s > 0 and fc > 0)) or
            (abs(s) >= abs(e) / 2.0)):
            s = m
            d = s
        else:
            e = d

        a = b
        fa = fb
        b = b + s
        fb = _safe_call(f, b, ctr)
        if math.isnan(fb):
            break
        if abs(fb) <= _TOL:
            return b, fb, True

    # Return best
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


def _newton_polish(f, x, fx, ctr, max_iter=10):
    """
    Final high-precision polish using Newton's method with finite-difference derivative.
    Returns (x, fx, success).
    """
    if math.isnan(fx):
        return x, fx, abs(fx) <= _TOL
    if abs(fx) <= _TOL:
        return x, fx, True

    h = 1e-8 * max(1.0, abs(x))
    for _ in range(max_iter):
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        if abs(fx) <= _TOL:
            return x, fx, True

        # Compute derivative via central difference
        fp = _safe_call(f, x + h, ctr)
        fm = _safe_call(f, x - h, ctr)
        if math.isnan(fp) or math.isnan(fm):
            break
        df = (fp - fm) / (2.0 * h)
        if abs(df) < 1e-15:
            break
        x_new = x - fx / df
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = _safe_call(f, x, ctr)
        if math.isnan(fx):
            break

    return x, fx, abs(fx) <= _TOL


def _taylor_model(f, x0, fx0, ctr, h_base=None):
    """
    Build a degree-4 Taylor polynomial around x0 using 5-point finite differences.
    Uses 5 additional f calls (at x0 ± h, x0 ± 2h).
    Returns list of coefficients [c0, c1, c2, c3, c4] for p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
    where t = x - x0, or None if insufficient data.
    """
    if h_base is None:
        h_base = max(1e-4, 0.01 * max(1.0, abs(x0)))
    
    # Evaluate f at 5 points: x0 - 2h, x0 - h, x0, x0 + h, x0 + 2h
    # We already have fx0 = f(x0)
    h = h_base
    x_m2 = x0 - 2*h
    x_m1 = x0 - h
    x_p1 = x0 + h
    x_p2 = x0 + 2*h
    
    fm2 = _safe_call(f, x_m2, ctr)
    fm1 = _safe_call(f, x_m1, ctr)
    fp1 = _safe_call(f, x_p1, ctr)
    fp2 = _safe_call(f, x_p2, ctr)
    
    vals = [fm2, fm1, fx0, fp1, fp2]
    if any(math.isnan(v) for v in vals):
        return None
    
    # Finite difference coefficients for derivatives using 5-point stencil
    # f'(x0) ≈ (-f(x0-2h) + 8f(x0-h) - 8f(x0+h) + f(x0+2h)) / (12h)
    # f''(x0) ≈ (f(x0-2h) - 2f(x0-h) + 2f(x0+h) - f(x0+2h)) / (2h^2)
    # f'''(x0) ≈ (f(x0-2h) - 2f(x0-h) + 2f(x0+h) - f(x0+2h)) / (2h^3)
    # Wait, let me use standard 5-point formulas:
    
    # First derivative (central, 5-point):
    # f' ≈ (-f[-2] + 8f[-1] - 8f[1] + f[2]) / (12h)
    f1 = (-fm2 + 8*fm1 - 8*fp1 + fp2) / (12.0 * h)
    
    # Second derivative (central, 5-point):
    # f'' ≈ (f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^2)
    h2 = h * h
    f2 = (fm2 - 2*fm1 + 2*fp1 - fp2) / (2.0 * h2)
    
    # Third derivative (central, 5-point):
    # f''' ≈ (f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^3)
    # Actually the standard 5-point third derivative is:
    # f''' ≈ (f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^3)
    # Let me verify: for a cubic, this should be exact.
    # Using Taylor: f(x+h) = f + h f' + h^2/2 f'' + h^3/6 f''' + h^4/24 f'''' + ...
    # The formula (f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^3) gives:
    # Numerator: [f - 2h f' + 2h^2 f'' - 4/3 h^3 f''' + ...] - 2[f - h f' + h^2/2 f'' - h^3/6 f''' + ...]
    #           + 2[f + h f' + h^2/2 f'' + h^3/6 f''' + ...] - [f + 2h f' + 2h^2 f'' + 4/3 h^3 f''' + ...]
    # = f - 2h f' + 2h^2 f'' - 4/3 h^3 f''' - 2f + 2h f' - h^2 f'' + h^3/3 f'''
    #   + 2f + 2h f' + h^2 f'' + h^3/3 f''' - f - 2h f' - 2h^2 f'' - 4/3 h^3 f'''
    # = (f - 2f + 2f - f) + (-2h f' + 2h f' + 2h f' - 2h f') + (2h^2 f'' - h^2 f'' + h^2 f'' - 2h^2 f'')
    #   + (-4/3 h^3 f''' + 1/3 h^3 f''' + 1/3 h^3 f''' - 4/3 h^3 f''')
    # = 0 + 0 + 0 + (-4/3 + 1/3 + 1/3 - 4/3) h^3 f'''
    # = (-8/3 + 2/3) h^3 f''' = -2 h^3 f'''
    # So (f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^3) = -f'''
    # We need the negative: f''' ≈ -(f[-2] - 2f[-1] + 2f[1] - f[2]) / (2h^3)
    h3 = h2 * h
    f3 = -(fm2 - 2*fm1 + 2*fp1 - fp2) / (2.0 * h3)
    
    # Fourth derivative (central, 5-point):
    # f'''' ≈ (f[-2] - 4f[-1] + 6f[0] - 4f[1] + f[2]) / h^4
    h4 = h2 * h2
    f4 = (fm2 - 4*fm1 + 6*fx0 - 4*fp1 + fp2) / h4
    
    # Taylor coefficients: c0 = f(x0), c1 = f'(x0), c2 = f''(x0)/2!, c3 = f'''(x0)/3!, c4 = f''''(x0)/4!
    c0 = fx0
    c1 = f1
    c2 = f2 / 2.0
    c3 = f3 / 6.0
    c4 = f4 / 24.0
    
    return [c0, c1, c2, c3, c4]


def _taylor_root_search(f, x0, fx0, ctr, max_candidates=20, h_base=None):
    """
    Use a Taylor polynomial model to find root candidates.
    Build model at x0, find roots of the polynomial via Newton iterations
    on the polynomial (no f calls), then verify with f.
    Returns (x, fx, success) for the best verified candidate, or None.
    """
    coeffs = _taylor_model(f, x0, fx0, ctr, h_base)
    if coeffs is None:
        return None
    
    c0, c1, c2, c3, c4 = coeffs
    
    def poly(t):
        return c0 + c1*t + c2*t*t + c3*t*t*t + c4*t*t*t*t
    
    def poly_deriv(t):
        return c1 + 2*c2*t + 3*c3*t*t + 4*c4*t*t*t
    
    # Find roots of the polynomial using Newton's method on the polynomial
    # Start with several initial guesses
    best_x = None
    best_fx = None
    best_abs = float('inf')
    
    # Initial guesses for t (offset from x0)
    if abs(c1) > 1e-15:
        t_newton = -c0 / c1  # First-order approximation
    else:
        t_newton = 0.1
    
    # Also try a few other starting points
    starts = [t_newton, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0]
    
    for t_start in starts:
        if ctr.n >= _CALL_BUDGET or _time_left() < 0.3:
            break
        
        t = t_start
        for _ in range(30):
            pt = poly(t)
            dpt = poly_deriv(t)
            if abs(dpt) < 1e-15:
                break
            t_new = t - pt / dpt
            if not math.isfinite(t_new):
                break
            if abs(t_new - t) < 1e-12 * max(1.0, abs(t)):
                t = t_new
                break
            t = t_new
        
        # Verify this candidate
        x_candidate = x0 + t
        fx_candidate = _safe_call(f, x_candidate, ctr)
        if math.isnan(fx_candidate):
            continue
        abs_fx = abs(fx_candidate)
        if abs_fx < best_abs:
            best_x = x_candidate
            best_fx = fx_candidate
            best_abs = abs_fx
        if abs_fx <= _TOL:
            return x_candidate, fx_candidate, True
    
    if best_x is not None:
        return best_x, best_fx, best_abs <= _TOL
    return None


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
        step *= 1.618  # golden ratio for steady expansion

    # If no bracket found, try a finer local scan for oscillatory functions
    # This is useful when the function oscillates rapidly
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
    """Try to bracket around x_start and solve with Brent."""
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

    # Strategy 1: Taylor series root search (uses ~5 extra calls to build model,
    # then Newton on polynomial is free, only verification calls f)
    taylor_result = _taylor_root_search(f, x0, fx0, ctr, max_candidates=15)
    if taylor_result is not None:
        x, fx, ok = taylor_result
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x, best_fx, best_abs = x, fx, abs(fx)
            if ok:
                return best_x

    # Strategy 2: bracket + Brent
    result = _try_bracket_and_brent(f, x0, fx0, ctr)
    if result is not None:
        x, fx = result
        if not math.isnan(fx):
            if abs(fx) < best_abs:
                best_x, best_fx, best_abs = x, fx, abs(fx)
            if abs(fx) <= _TOL:
                return best_x

    # Strategy 3: secant
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=30)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 4: try offsets
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
        # Try bracket from this point
        result = _try_bracket_and_brent(f, x_try, fx_try, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 5: secant from best
    if best_abs > _TOL and not math.isnan(best_fx):
        x_s, fx_s, ok = _secant(f, best_x, best_fx, ctr, max_iter=20)
        if not math.isnan(fx_s):
            if abs(fx_s) < best_abs:
                best_x, best_fx, best_abs = x_s, fx_s, abs(fx_s)
            if ok:
                return best_x

    # Strategy 6: try to bracket from best point
    if best_abs > _TOL and not math.isnan(best_fx):
        result = _try_bracket_and_brent(f, best_x, best_fx, ctr)
        if result is not None:
            x, fx = result
            if not math.isnan(fx):
                if abs(fx) < best_abs:
                    best_x, best_fx, best_abs = x, fx, abs(fx)
                if abs(fx) <= _TOL:
                    return best_x

    # Strategy 7: final Newton polish for extra precision
    if best_abs > _TOL and not math.isnan(best_fx) and _time_left() > 0.5:
        x_p, fx_p, ok_p = _newton_polish(f, best_x, best_fx, ctr, max_iter=10)
        if not math.isnan(fx_p):
            if abs(fx_p) < best_abs:
                best_x, best_fx, best_abs = x_p, fx_p, abs(fx_p)
            if ok_p:
                return best_x

    return best_x
