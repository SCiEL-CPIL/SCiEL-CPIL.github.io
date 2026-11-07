"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Quick check at x0.
  2. Adaptive bracket search using geometric expansion with sign checks.
  3. Brent's method (bisection + secant + inverse quadratic interpolation)
     for fast, robust convergence within the bracket.
  4. Fallback to secant/Newton if no bracket found.

Minimizes function calls while maintaining robustness for oscillatory f.
"""
import time
import math

_START_TIME = None
_DEADLINE = 145.0  # Safety margin before 150s limit


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def _is_finite(x):
    return math.isfinite(x)


def _brent_method(f, a, b, fa, fb, max_iter=80):
    """
    Brent's method for root finding within [a, b] where f(a)*f(b) < 0.
    Uses bisection, secant, and inverse quadratic interpolation.
    Returns (x, fx) where x is the best root estimate.
    """
    # Ensure |f(a)| <= |f(b)| for the algorithm's convention
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(max_iter):
        if _elapsed() > _DEADLINE:
            break

        # Check convergence
        if abs(fb) <= 1e-10:
            return b, fb
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

        # Inverse quadratic interpolation
        do_iq = False
        if a != b and b != c:
            # Inverse quadratic interpolation
            try:
                r1 = (a / (b - a)) * (fb - fc)
                r2 = (b / (c - b)) * (fb - fc)
                # Standard Brent formula
                num = (a * (fb - fc) * (fb - fa) / ((b - a) * (c - b))
                       - b * (fa - fc) * (fb - fa) / ((c - b) * (a - c))
                       + c * (fa - fb) * (fb - fc) / ((a - c) * (b - a)))
                # Simpler: use the standard formulation
                # s = (a*fb*fc/(fb-fa)/(fb-fc) + b*fa*fc/(fa-fb)/(fa-fc) + c*fa*fb/(fa-fc)/(fb-fa))
                # Let's use a cleaner formulation
                denom = (fa - fb) * (fa - fc) * (fb - fc)
                if abs(denom) > 1e-30:
                    s = (a * fb * fc / ((fb - fa) * (fb - fc))
                         + b * fa * fc / ((fa - fb) * (fa - fc))
                         + c * fa * fb / ((fa - fc) * (fb - fa)))
                    do_iq = True
                else:
                    do_iq = False
            except (ZeroDivisionError, OverflowError):
                do_iq = False
        else:
            do_iq = False

        if do_iq:
            # Check if s is acceptable
            if not ((min(a, b, c) < s < max(a, b, c)) or (s < min(a, b, c)) or (s > max(a, b, c))):
                # s must be between a and b (or in the bracket)
                pass
            if not ((2.0 * s - (a + b)) < (abs(b - c) * 0.9) and
                    (s - a) > (abs(b - c) * 0.1)):
                do_iq = False

        if do_iq:
            x = s
        else:
            # Secant method
            if abs(e) > 1e-30:
                try:
                    x = b - (b - a) * fb / (fb - fa)
                except (ZeroDivisionError, OverflowError):
                    x = (a + b) / 2.0
            else:
                x = (a + b) / 2.0
                e = d
                d = x - b
                continue

        # Check if x is too close to boundaries
        if not ((x - a) < (b - x) * 0.9 and (x - a) > (b - a) * 0.1):
            x = (a + b) / 2.0
            d = x - b
            e = d
            continue

        e = d
        d = x - b

        try:
            fx = f(x)
        except:
            # Fallback to bisection
            x = (a + b) / 2.0
            try:
                fx = f(x)
            except:
                break

        if fx is None or not _is_finite(fx):
            break

        if abs(fx) <= 1e-10:
            return x, fx

        # Update bracket
        if fa * fx < 0:
            c = b
            fc = fb
            b = x
            fb = fx
        else:
            c = a
            fc = fa
            a = x
            fa = fx

        # Maintain |f(a)| <= |f(b)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Return best point
    if abs(fa) < abs(fb):
        return a, fa
    else:
        return b, fb


def _illinois_method(f, a, b, fa, fb, max_iter=100):
    """
    Illinois method for root finding within [a, b] where f(a)*f(b) < 0.
    Uses inverse linear interpolation with bisection fallback.
    Returns (x, fx) where x is the best root estimate.
    """
    left = a
    fleft = fa
    right = b
    fright = fb
    
    # Ensure f(left) * f(right) < 0
    if fleft * fright > 0:
        return (a, fa)
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        
        # Check if we're close enough
        if abs(fleft) <= 1e-10:
            return left, fleft
        if abs(fright) <= 1e-10:
            return right, fright
        
        # Illinois interpolation: 
        # x_new = (left * fright - right * fleft) / (fright - fleft)
        # This is inverse linear interpolation.
        denom = fright - fleft
        if abs(denom) > 1e-30:
            x_new = (left * fright - right * fleft) / denom
        else:
            x_new = (left + right) / 2.0
        
        # If x_new is not strictly between left and right, use bisection.
        if not (left < x_new < right):
            x_new = (left + right) / 2.0
        
        try:
            f_new = f(x_new)
        except:
            f_new = None
        
        if f_new is None or not _is_finite(f_new):
            # Fallback to bisection
            x_new = (left + right) / 2.0
            try:
                f_new = f(x_new)
            except:
                f_new = None
            if f_new is None or not _is_finite(f_new):
                break
        
        if abs(f_new) <= 1e-8:
            return x_new, f_new
        
        if f_new * fleft < 0:
            right = x_new
            fright = f_new
        else:
            left = x_new
            fleft = f_new
        
        # Check convergence
        if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
            break
    
    # Return the point with smallest |f|
    if abs(fleft) < abs(fright):
        return left, fleft
    else:
        return right, fright


def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Returns a float x.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0

    # Quick check at x0
    try:
        f0 = f(x0)
    except:
        f0 = float('inf')

    if f0 is not None and _is_finite(f0) and abs(f0) <= 1e-8:
        return x0

    # Phase 1: Find a bracket [a, b] with f(a)*f(b) < 0
    # Use adaptive geometric search from x0
    best_x = x0
    best_f = f0 if _is_finite(f0) else float('inf')

    a = None
    fa = None
    b = None
    fb = None

    # First pass: try a set of offsets from x0
    offsets = []
    for s in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
        offsets.append(s)
        offsets.append(-s)

    points = [(x0, f0)]  # (x, fx) pairs

    for dx in offsets:
        if _elapsed() > _DEADLINE:
            break
        x_try = x0 + dx
        try:
            fx_try = f(x_try)
        except:
            continue
        if not _is_finite(fx_try):
            continue

        if abs(fx_try) < abs(best_f):
            best_f = fx_try
            best_x = x_try

        if abs(fx_try) <= 1e-8:
            return x_try

        # Check for sign change with previous point
        if points:
            px, pfx = points[-1]
            if pfx * fx_try < 0:
                a, fa = px, pfx
                b, fb = x_try, fx_try
                break

        points.append((x_try, fx_try))

        # Also check against x0
        if f0 is not None and _is_finite(f0) and f0 * fx_try < 0:
            if abs(f0) < abs(fx_try):
                a, fa = x0, f0
                b, fb = x_try, fx_try
            else:
                a, fa = x_try, fx_try
                b, fb = x0, f0
            break

    # If no bracket found, try geometric expansion more aggressively
    if b is None:
        for scale in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]:
            if _elapsed() > _DEADLINE:
                break
            for sign in [1, -1]:
                if _elapsed() > _DEADLINE:
                    break
                x_try = x0 + sign * scale * max(1.0, abs(x0))
                try:
                    fx_try = f(x_try)
                except:
                    continue
                if not _is_finite(fx_try):
                    continue

                if abs(fx_try) < abs(best_f):
                    best_f = fx_try
                    best_x = x_try

                if abs(fx_try) <= 1e-8:
                    return x_try

                # Check against x0
                if f0 is not None and _is_finite(f0) and f0 * fx_try < 0:
                    if abs(f0) < abs(fx_try):
                        a, fa = x0, f0
                        b, fb = x_try, fx_try
                    else:
                        a, fa = x_try, fx_try
                        b, fb = x0, f0
                    break

                # Check against best point
                if best_f is not None and _is_finite(best_f) and best_f * fx_try < 0:
                    if abs(best_f) < abs(fx_try):
                        a, fa = best_x, best_f
                        b, fb = x_try, fx_try
                    else:
                        a, fa = x_try, fx_try
                        b, fb = best_x, best_f
                    break
            if b is not None:
                break

    # Phase 2: If we have a bracket, use Brent's method
    if b is not None and fa is not None and fb is not None:
        if fa * fb < 0:
            # Try Brent's method first
            x_root, f_root = _brent_method(f, a, b, fa, fb)
            if _is_finite(x_root) and _is_finite(f_root) and abs(f_root) <= 1e-8:
                return x_root
            
            # If Brent's method didn't converge, try Illinois method
            x_root2, f_root2 = _illinois_method(f, a, b, fa, fb)
            if _is_finite(x_root2) and _is_finite(f_root2) and abs(f_root2) <= 1e-8:
                return x_root2
            
            # Return the best we found
            if _is_finite(x_root) and _is_finite(f_root):
                if abs(f_root) < abs(best_f):
                    best_x = x_root
                    best_f = f_root
                return x_root
            if _is_finite(x_root2) and _is_finite(f_root2):
                if abs(f_root2) < abs(best_f):
                    best_x = x_root2
                    best_f = f_root2
                return x_root2

    # Phase 3: No bracket found - use secant/Newton-like local search
    start_points = [x0, x0 + 0.1, x0 - 0.1, x0 + 0.5, x0 - 0.5, x0 + 1.0, x0 - 1.0]

    for sx in start_points:
        if _elapsed() > _DEADLINE:
            break
        try:
            fx = f(sx)
        except:
            continue
        if not _is_finite(fx):
            continue

        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = sx

        if abs(fx) <= 1e-8:
            return sx

        x_prev = sx
        fx_prev = fx
        h = 1e-6 * max(1.0, abs(sx))
        x_curr = sx + h
        try:
            fx_curr = f(x_curr)
        except:
            continue
        if not _is_finite(fx_curr):
            continue

        if abs(fx_curr) < abs(best_f):
            best_f = fx_curr
            best_x = x_curr

        if abs(fx_curr) <= 1e-8:
            return x_curr

        for _ in range(50):
            if _elapsed() > _DEADLINE:
                break

            if abs(fx_curr) <= 1e-8:
                return x_curr

            if abs(fx_curr - fx_prev) > 1e-30:
                x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                break

            if not _is_finite(x_new):
                break

            if abs(x_new - x_curr) > 2.0 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr) * (1 if x_new > x_curr else -1)

            try:
                fx_new = f(x_new)
            except:
                break

            if fx_new is None or not _is_finite(fx_new):
                break

            if abs(fx_new) < abs(best_f):
                best_f = fx_new
                best_x = x_new

            if abs(fx_new) <= 1e-8:
                return x_new

            if abs(fx_new) < abs(fx_curr):
                x_prev = x_curr
                fx_prev = fx_curr
                x_curr = x_new
                fx_curr = fx_new
            else:
                x_new = x_curr - 0.1 * (x_curr - x_prev) * fx_curr / (fx_curr - fx_prev)
                if not _is_finite(x_new):
                    break
                try:
                    fx_new = f(x_new)
                except:
                    break
                if fx_new is None or not _is_finite(fx_new):
                    break
                if abs(fx_new) < abs(best_f):
                    best_f = fx_new
                    best_x = x_new
                if abs(fx_new) <= 1e-8:
                    return x_new
                if abs(fx_new) < abs(fx_curr):
                    x_prev = x_curr
                    fx_prev = fx_curr
                    x_curr = x_new
                    fx_curr = fx_new
                else:
                    break

    return best_x
