"""Root finding for ill-conditioned problems with multiple strategies."""
import time
import math

_TOL = 1e-8
_MAX_EVALS = 500
# Safety margin well under the 150s kill
_DEADLINE_SECONDS = 145.0


def _eval_count_wrapper(f):
    """Wrap f to count evaluations."""
    state = {'count': 0}
    def wrapped(x):
        state['count'] += 1
        return f(x)
    return wrapped, state


def _is_finite(x):
    return math.isfinite(x)


def _safe_f(f, x):
    """Safely evaluate f, returning inf if it fails."""
    try:
        val = f(x)
        if not _is_finite(val):
            return float('inf')
        return val
    except (OverflowError, ZeroDivisionError, ValueError):
        return float('inf')


def _newton_with_bisection(f, x0, max_evals=200, tol=1e-8):
    """Newton's method with bisection fallback for robustness."""
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x, 1
    
    # Try to establish a bracket
    a, b = None, None
    fa, fb = None, None
    
    # Check if x0 is close to root
    if abs(fx) <= tol:
        return x, 1
    
    # Try Newton steps with safeguards
    x_new = x
    fx_new = fx
    evals = 1
    
    # First, try pure Newton for a few steps
    for i in range(20):
        if not _is_finite(x) or not _is_finite(fx):
            break
        if abs(fx) <= tol:
            return x, evals
        
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        fp = _safe_f(f, x + h)
        fm = _safe_f(f, x - h)
        evals += 2
        
        if not _is_finite(fp) or not _is_finite(fm):
            # Try forward difference
            fp = _safe_f(f, x + h)
            evals += 1
            if not _is_finite(fp):
                break
            d = (fp - fx) / h
        else:
            d = (fp - fm) / (2.0 * h)
        
        if d == 0 or not _is_finite(d):
            break
        
        x_new = x - fx / d
        
        if not _is_finite(x_new):
            break
        
        # Check for progress
        if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
            break
        
        fx_new = _safe_f(f, x_new)
        evals += 1
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) <= tol:
            return x_new, evals
        
        # If Newton step is too large, use bisection-like damping
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
            fx_new = _safe_f(f, x_new)
            evals += 1
            if not _is_finite(fx_new):
                break
            if abs(fx_new) <= tol:
                return x_new, evals
        
        x, fx = x_new, fx_new
    
    # If Newton didn't converge, try to find a bracket and use bisection
    if abs(fx) <= tol:
        return x, evals
    
    # Try to find a bracket around x
    step = max(1.0, abs(x)) * 0.1
    for i in range(50):
        if evals >= max_evals:
            break
        
        left = x - step
        right = x + step
        
        fl = _safe_f(f, left)
        fr = _safe_f(f, right)
        evals += 2
        
        if not _is_finite(fl) or not _is_finite(fr):
            step *= 2
            continue
        
        if abs(fl) <= tol:
            return left, evals
        if abs(fr) <= tol:
            return right, evals
        
        if fl * fr < 0:
            # Found a bracket
            a, b = left, right
            fa, fb = fl, fr
            break
        
        step *= 2
        if step > 1e10:
            break
    
    if a is not None and b is not None:
        # Bisection
        for i in range(60):
            if evals >= max_evals:
                break
            mid = (a + b) / 2.0
            fm = _safe_f(f, mid)
            evals += 1
            if not _is_finite(fm):
                break
            if abs(fm) <= tol:
                return mid, evals
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
        return (a + b) / 2.0, evals
    
    return x, evals


def _secant_method(f, x0, max_evals=200, tol=1e-8):
    """Secant method with safeguards."""
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x, 1
    
    # Initial second point
    h = 1e-6 * max(1.0, abs(x))
    x1 = x + h
    fx1 = _safe_f(f, x1)
    evals = 2
    
    if not _is_finite(fx1):
        x1 = x - h
        fx1 = _safe_f(f, x1)
        evals = 3
    
    if not _is_finite(fx1):
        return x, evals
    
    for i in range(100):
        if evals >= max_evals:
            break
        
        if fx1 == fx:
            break
        
        x2 = x1 - fx1 * (x1 - x) / (fx1 - fx)
        
        if not _is_finite(x2):
            break
        
        # Damping if step is too large
        if abs(x2 - x1) > 10.0 * max(1.0, abs(x1)):
            x2 = x1 + 0.5 * (x2 - x1)
        
        fx2 = _safe_f(f, x2)
        evals += 1
        
        if not _is_finite(fx2):
            break
        
        if abs(fx2) <= tol:
            return x2, evals
        
        x, fx = x1, fx1
        x1, fx1 = x2, fx2
    
    # Return best estimate
    if abs(fx) < abs(fx1):
        return x, evals
    return x1, evals


def _brent_like(f, x0, max_evals=200, tol=1e-8):
    """A simplified Brent-like method combining secant, bisection, and inverse quadratic interpolation."""
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x, 1
    
    # Establish bracket
    a, b = None, None
    fa, fb = None, None
    
    step = max(1.0, abs(x)) * 0.1
    for i in range(30):
        if max_evals <= 10:
            break
        
        left = x - step
        right = x + step
        
        fl = _safe_f(f, left)
        fr = _safe_f(f, right)
        max_evals -= 2
        
        if not _is_finite(fl) or not _is_finite(fr):
            step *= 2
            continue
        
        if abs(fl) <= tol:
            return left, _MAX_EVALS - max_evals
        if abs(fr) <= tol:
            return right, _MAX_EVALS - max_evals
        
        if fl * fr < 0:
            a, b = left, right
            fa, fb = fl, fr
            break
        
        step *= 2
        if step > 1e10:
            break
    
    if a is None:
        # No bracket found, fall back to secant
        return _secant_method(f, x0, max_evals, tol)
    
    # Now use a combination of methods
    c = a
    fc = fa
    d = a - b
    
    for i in range(100):
        if max_evals <= 0:
            break
        
        if fa == fb or abs(b - a) < 1e-15:
            break
        
        # Inverse quadratic interpolation
        try:
            num1 = b * (fb - fa)
            num2 = a * (fb - fc)
            den = fa - 2.0 * fb + fc
            if den != 0:
                xq = a - den * (num1 * num2) / (2.0 * (fa - fb) * (fa - fc) * (fb - fc)) if False else \
                     b - (b - a) * (fb * (fb - fc)) / ((fb - fa) * (fb - fc)) + (b - c) * (fb * (fb - fa)) / ((fb - fc) * (fb - fc)) if False else \
                     b - (b - a) * (fb * (fb - fc)) / ((fb - fa) * (fb - fc))
            else:
                xq = None
        except (ZeroDivisionError, OverflowError):
            xq = None
        
        # Secant
        xs = b - fb * (b - a) / (fb - fa)
        
        # Choose which to use
        if xq is not None and _is_finite(xq):
            # Check if xq is in the bracket
            if (min(a, b) < xq < max(a, b)) and abs(xq - b) < 0.75 * abs(b - c):
                x_new = xq
            else:
                x_new = xs
        else:
            x_new = xs
        
        if not _is_finite(x_new):
            x_new = (a + b) / 2.0
        
        fx_new = _safe_f(f, x_new)
        max_evals -= 1
        
        if not _is_finite(fx_new):
            x_new = (a + b) / 2.0
            fx_new = _safe_f(f, x_new)
            max_evals -= 1
        
        if not _is_finite(fx_new):
            break
        
        if abs(fx_new) <= tol:
            return x_new, _MAX_EVALS - max_evals
        
        c, fc = b, fb
        b, fb = x_new, fx_new
        
        if fa * fb < 0:
            a, fa = c, fc
        else:
            a, fa = b, fb
        
        # Ensure b is the better root
        if abs(fa) < abs(fb):
            a, fa, b, fb = b, fb, a, fa
    
    # Return best
    if abs(fa) < abs(fb):
        return a, _MAX_EVALS - max_evals
    return b, _MAX_EVALS - max_evals


def find_root(f, x0):
    """Find root of f starting from x0. Returns x with |f(x)| <= 1e-8."""
    start_time = time.monotonic()
    
    # Wrap f to count evaluations
    wrapped_f, state = _eval_count_wrapper(f)
    
    def f_counted(x):
        if time.monotonic() - start_time > _DEADLINE_SECONDS:
            return float('inf')
        return wrapped_f(x)
    
    # Try multiple strategies and return the best
    best_x = float(x0)
    best_fx = abs(f_counted(best_x))
    best_evals = 1
    
    if best_fx <= _TOL:
        return best_x
    
    # Strategy 1: Newton with bisection fallback
    x1, evals1 = _newton_with_bisection(f_counted, x0, max_evals=150, tol=_TOL)
    if _is_finite(x1):
        fx1 = abs(f_counted(x1))
        if fx1 < best_fx:
            best_x, best_fx, best_evals = x1, fx1, state['count']
    
    if best_fx <= _TOL:
        return best_x
    
    # Check time
    if time.monotonic() - start_time > _DEADLINE_SECONDS - 5:
        return best_x
    
    # Strategy 2: Secant method
    x2, evals2 = _secant_method(f_counted, x0, max_evals=100, tol=_TOL)
    if _is_finite(x2):
        fx2 = abs(f_counted(x2))
        if fx2 < best_fx:
            best_x, best_fx, best_evals = x2, fx2, state['count']
    
    if best_fx <= _TOL:
        return best_x
    
    # Check time
    if time.monotonic() - start_time > _DEADLINE_SECONDS - 5:
        return best_x
    
    # Strategy 3: Brent-like method
    x3, evals3 = _brent_like(f_counted, x0, max_evals=100, tol=_TOL)
    if _is_finite(x3):
        fx3 = abs(f_counted(x3))
        if fx3 < best_fx:
            best_x, best_fx, best_evals = x3, fx3, state['count']
    
    if best_fx <= _TOL:
        return best_x
    
    # If we still haven't found a root, try a local search around best_x
    if time.monotonic() - start_time < _DEADLINE_SECONDS - 10 and state['count'] < _MAX_EVALS:
        # Try small perturbations
        for scale in [1e-4, 1e-3, 1e-2, 1e-1]:
            if state['count'] >= _MAX_EVALS or time.monotonic() - start_time > _DEADLINE_SECONDS - 5:
                break
            for dx in [-scale * max(1.0, abs(best_x)), scale * max(1.0, abs(best_x))]:
                if state['count'] >= _MAX_EVALS or time.monotonic() - start_time > _DEADLINE_SECONDS - 5:
                    break
                x_try = best_x + dx
                if _is_finite(x_try):
                    fx_try = abs(f_counted(x_try))
                    if fx_try < best_fx:
                        best_x, best_fx = x_try, fx_try
                        if best_fx <= _TOL:
                            return best_x
    
    return best_x
