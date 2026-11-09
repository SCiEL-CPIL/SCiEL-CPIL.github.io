"""Direct root finder for smooth polynomials using interpolation + safeguarded Newton."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 145.0
_SAFETY = 5.0
_CALLS_PER_EQ = 150  # Per-equation call budget to keep mean low


def _time_left():
    return _WALL_LIMIT - _SAFETY - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _poly_root(c, x0):
    """Find the root of a cubic/quadratic/linear polynomial (coeffs high-to-low)
    closest to x0, using numpy.roots."""
    import numpy as np
    # Trim leading zeros
    while len(c) > 1 and abs(c[0]) < 1e-30:
        c = c[1:]
    if len(c) <= 1:
        return None
    if len(c) == 2:
        # ax + b = 0
        if abs(c[0]) < 1e-30:
            return None
        return -c[1] / c[0]
    # Use numpy for cubic/quadratic
    try:
        roots = np.roots(c)
    except Exception:
        return None
    # Pick the real root closest to x0
    best = None
    best_d = float('inf')
    for r in roots:
        if abs(r.imag) < 1e-10 * max(1.0, abs(r.real)):
            d = abs(r.real - x0)
            if d < best_d:
                best_d = d
                best = r.real
    return best


def _inverse_quadratic_interpolation(xs, fs):
    """Given 3 points (x, f), find x where the inverse quadratic (x as function of f) is 0.
    Uses Lagrange interpolation on x(f) with f=0.
    Returns the x value or None if ill-conditioned."""
    x0, x1, x2 = xs
    f0, f1, f2 = fs
    # Avoid division by near-zero differences
    if (abs(f0 - f1) < 1e-30 or abs(f1 - f2) < 1e-30 or abs(f0 - f2) < 1e-30):
        return None
    # Lagrange basis at f=0
    L0 = ((0 - f1) * (0 - f2)) / ((f0 - f1) * (f0 - f2))
    L1 = ((0 - f0) * (0 - f2)) / ((f1 - f0) * (f1 - f2))
    L2 = ((0 - f0) * (0 - f1)) / ((f2 - f0) * (f2 - f1))
    x_new = L0 * x0 + L1 * x1 + L2 * x2
    if not _is_finite(x_new):
        return None
    return x_new


def _cubic_interpolation_root(xs, fs, x0):
    """Fit a cubic polynomial through 4 points and find the root closest to x0."""
    x0v, x1v, x2v, x3v = xs
    f0v, f1v, f2v, f3v = fs
    # Build Vandermonde-like system for Newton form coefficients
    # p(x) = a0 + a1*(x-x0) + a2*(x-x0)(x-x1) + a3*(x-x0)(x-x1)(x-x2)
    # Solve for a0, a1, a2, a3
    # a0 = f0
    a0 = f0v
    # a1 = (f1 - f0) / (x1 - x0)
    if abs(x1v - x0v) < 1e-30:
        return None
    a1 = (f1v - f0v) / (x1v - x0v)
    # a2 = ((f2 - f0)/(x2-x0) - a1) / (x2 - x1)
    if abs(x2v - x0v) < 1e-30 or abs(x2v - x1v) < 1e-30:
        return None
    a2 = ((f2v - f0v) / (x2v - x0v) - a1) / (x2v - x1v)
    # a3 = ((f3 - f0)/(x3-x0) - a1 - a2*(x3-x1)) / ((x3-x1)*(x3-x2))
    if abs(x3v - x0v) < 1e-30 or abs(x3v - x1v) < 1e-30 or abs(x3v - x2v) < 1e-30:
        return None
    a3 = ((f3v - f0v) / (x3v - x0v) - a1 - a2 * (x3v - x1v)) / ((x3v - x1v) * (x3v - x2v))
    
    # Convert to standard form: p(x) = c3*x^3 + c2*x^2 + c1*x + c0
    # Expand Newton form
    # (x-x0) = x - x0
    # (x-x0)(x-x1) = x^2 - (x0+x1)x + x0*x1
    # (x-x0)(x-x1)(x-x2) = x^3 - (x0+x1+x2)x^2 + (x0*x1+x0*x2+x1*x2)x - x0*x1*x2
    c3 = a3
    c2 = a2 - a3 * (x0v + x1v + x2v)
    c1 = a1 - a2 * (x0v + x1v) + a3 * (x0v * x1v + x0v * x2v + x1v * x2v)
    c0 = a0 - a1 * x0v + a2 * x0v * x1v - a3 * x0v * x1v * x2v
    
    return _poly_root([c3, c2, c1, c0], x0)


def find_root(f, x0):
    tol = 1e-8
    max_calls = _CALLS_PER_EQ
    calls = 0

    def ceval(x):
        nonlocal calls
        if calls >= max_calls:
            return None, False
        calls += 1
        try:
            val = f(x)
        except Exception:
            return None, False
        if not _is_finite(val):
            return None, False
        return val, True

    # --- Phase 0: initial evaluation at x0 ---
    x = float(x0)
    fx, ok = ceval(x)
    if not ok:
        for dx in (1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0):
            if calls >= max_calls:
                break
            xt = x + dx
            fx, ok = ceval(xt)
            if ok:
                x = xt
                break
        if not ok:
            return x

    if abs(fx) <= tol:
        return x

    # Collect evaluation history for interpolation
    history = [(x, fx)]
    best_x = x
    best_f = abs(fx)

    # --- Phase 1: Build up 3 points for secant/inverse-quadratic ---
    # Use secant-style steps to get diverse points quickly
    h0 = 1e-2 * max(1.0, abs(x))
    for dx in (h0, -h0, 5*h0, -5*h0):
        if len(history) >= 3 or calls >= max_calls:
            break
        xt = x + dx
        ft, okt = ceval(xt)
        if okt:
            # Check for duplicate x
            dup = False
            for hx, hf in history:
                if abs(hx - xt) < 1e-14 * max(1.0, abs(hx)):
                    dup = True
                    break
            if not dup:
                history.append((xt, ft))
                if abs(ft) < best_f:
                    best_f = abs(ft)
                    best_x = xt
                if abs(ft) <= tol:
                    return xt

    if best_f <= tol:
        return best_x

    # --- Phase 2: Iterative refinement using secant + inverse quadratic ---
    # Prioritize secant (1 call/iter) with inverse quadratic when beneficial
    for iteration in range(100):
        if calls >= max_calls or _time_left() < 0:
            break
        
        # Decide which method to use based on available history
        x_new = None
        
        if len(history) >= 3:
            # Try inverse quadratic first if points are well-conditioned
            xs = [h[0] for h in history[-3:]]
            fs = [h[1] for h in history[-3:]]
            x_new = _inverse_quadratic_interpolation(xs, fs)
        
        if x_new is None and len(history) >= 2:
            # Secant
            x1, f1 = history[-2]
            x2, f2 = history[-1]
            denom = f2 - f1
            if abs(denom) > 1e-18:
                x_new = x2 - f2 * (x2 - x1) / denom
        
        if x_new is None:
            # Fallback: small step from best point
            x_new = best_x + 0.1 * max(1.0, abs(best_x))
        
        if not _is_finite(x_new):
            break
        
        # Divergence guard
        if abs(x_new) > 1e12 * max(1.0, abs(best_x)):
            x_new = best_x + 0.5 * (history[-1][0] - best_x)
            if not _is_finite(x_new):
                break
        
        fx_new, ok = ceval(x_new)
        if not ok:
            break
        
        history.append((x_new, fx_new))
        
        if abs(fx_new) < best_f:
            best_f = abs(fx_new)
            best_x = x_new
        
        if abs(fx_new) <= tol:
            return x_new
        
        # Keep history manageable (last 4 points)
        if len(history) > 4:
            history = history[-4:]
        
        # Check for stagnation
        if len(history) >= 2:
            if abs(history[-1][0] - history[-2][0]) < 1e-14 * max(1.0, abs(history[-1][0])):
                break

    if best_f <= tol:
        return best_x

    # --- Phase 3: Brief bisection fallback if bracket can be found quickly ---
    if calls < max_calls - 5:
        x_lo, x_hi = best_x, best_x
        f_lo = None
        # Try to find a sign change
        step = 0.1 * max(1.0, abs(best_x))
        bracketed = False
        for _ in range(8):
            if calls >= max_calls or _time_left() < 0:
                break
            x_try = best_x - step
            f_try, ok = ceval(x_try)
            if ok:
                if f_lo is None:
                    f_lo = f(best_x) if calls < max_calls else None
                if f_lo is not None and f_lo * f_try < 0:
                    x_hi = best_x
                    x_lo = x_try
                    f_lo = f_try
                    bracketed = True
                    break
            step *= 2.0

        if bracketed:
            for _ in range(20):
                if calls >= max_calls or _time_left() < 0:
                    break
                x_mid = 0.5 * (x_lo + x_hi)
                f_mid, ok = ceval(x_mid)
                if not ok:
                    break
                if abs(f_mid) <= tol:
                    return x_mid
                if abs(f_mid) < best_f:
                    best_f = abs(f_mid)
                    best_x = x_mid
                # Need f at x_hi for comparison
                f_hi, ok = ceval(x_hi)
                if not ok:
                    break
                if f_lo * f_mid <= 0:
                    x_hi = x_mid
                else:
                    x_lo = x_mid
                    f_lo = f_mid

    return best_x
