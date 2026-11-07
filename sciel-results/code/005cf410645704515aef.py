"""Derivative-free root finder: Secant + Brent's safeguarded inverse interpolation."""
import math
import time

_DEADLINE = time.monotonic() + 140.0  # safety margin before 150s kill


def _check_deadline():
    return time.monotonic() < _DEADLINE


def find_root(f, x0):
    x = float(x0)
    best_x = x
    best_fx = None

    # Evaluate initial
    try:
        fx = f(x)
    except Exception:
        return x
    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_fx = fx
    best_x = x

    # Phase 1: Secant method (derivative-free, 1 call per iteration after initial)
    # Use x0 and x0+step as the two initial points
    step = max(1e-4, abs(x) * 1e-4)
    x1 = x
    f1 = fx
    x2 = x + step
    try:
        f2 = f(x2)
    except Exception:
        return best_x
    if not math.isfinite(f2):
        return best_x

    if abs(f2) < abs(best_fx):
        best_x = x2
        best_fx = f2

    if abs(f2) <= 1e-8:
        return x2

    # Secant iterations: reuse the two most recent evaluations
    converged = False
    for i in range(30):
        if not _check_deadline():
            break
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2

        # Secant step
        denom = f2 - f1
        if abs(denom) < 1e-300:
            break

        x_new = x2 - f2 * (x2 - x1) / denom

        # Damping: limit step size
        if abs(x_new - x2) > 5.0 * max(1.0, abs(x2)):
            x_new = x2 + 5.0 * max(1.0, abs(x2)) * (1 if x_new > x2 else -1)

        if not math.isfinite(x_new):
            break

        try:
            f_new = f(x_new)
        except Exception:
            break
        if not math.isfinite(f_new):
            break

        if abs(f_new) < abs(best_fx):
            best_x = x_new
            best_fx = f_new

        if abs(f_new) <= 1e-8:
            return x_new

        # Shift the window: keep the two most recent points
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

    if abs(f1) <= 1e-8:
        return x1
    if abs(f2) <= 1e-8:
        return x2

    # Phase 2: If secant didn't converge, try to find a bracket and use Brent's method
    if abs(best_fx) > 1e-8 and _check_deadline():
        result = _brent_solve(f, best_x, best_fx, x1, f1, x2, f2)
        if result is not None:
            try:
                fx_check = f(result)
                if math.isfinite(fx_check) and abs(fx_check) <= 1e-8:
                    return result
                if math.isfinite(fx_check) and abs(fx_check) < abs(best_fx):
                    best_x = result
                    best_fx = fx_check
            except Exception:
                pass

    return best_x


def _brent_solve(f, x_best, fx_best, x1, f1, x2, f2):
    """Brent's method with safeguarded inverse interpolation.
    First tries to establish a bracket from available points, then applies the algorithm."""
    if not _check_deadline():
        return None

    # Try to find a bracket using existing evaluations and minimal new ones
    # Candidates for bracket endpoints: x1, x2, x_best
    candidates = []
    for (xi, fi) in [(x1, f1), (x2, f2), (x_best, fx_best)]:
        if math.isfinite(xi) and math.isfinite(fi) and abs(fi) > 0:
            candidates.append((xi, fi))

    if len(candidates) < 2:
        return None

    # Check if any pair of existing candidates forms a bracket
    bracket_a, bracket_fa, bracket_b, bracket_fb = None, None, None, None
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            xa, fa = candidates[i]
            xb, fb = candidates[j]
            if fa * fb < 0:
                bracket_a, bracket_fa = xa, fa
                bracket_b, bracket_fb = xb, fb
                break
        if bracket_a is not None:
            break

    # If no bracket found, try to expand from the best point
    if bracket_a is None:
        a = x_best
        fa = fx_best
        step = 1.0
        found = False
        for _ in range(20):
            if not _check_deadline():
                break
            b = a + step
            try:
                fb = f(b)
            except Exception:
                step *= 2.0
                continue
            if not math.isfinite(fb):
                step *= 2.0
                continue
            if fa * fb < 0:
                bracket_a, bracket_fa = a, fa
                bracket_b, bracket_fb = b, fb
                found = True
                break
            b = a - step
            try:
                fb = f(b)
            except Exception:
                step *= 2.0
                continue
            if not math.isfinite(fb):
                step *= 2.0
                continue
            if fa * fb < 0:
                bracket_a, bracket_fa = a, fa
                bracket_b, bracket_fb = b, fb
                found = True
                break
            step *= 2.0
        if not found:
            return None

    # Now apply Brent's method on [bracket_a, bracket_b]
    a, fa = bracket_a, bracket_fa
    b, fb = bracket_b, bracket_fb
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # Ensure |fa| < |fb| so that c becomes the previous b
    c, fc = a, fa
    a, fa = b, fb
    b, fb = c, fc

    # Now |fa| > |fb|, c is the point with smaller |f|
    # Actually in Brent's: a and b bracket, with |f(a)| > |f(b)|
    # Let me re-initialize properly:
    # We want: f(a) and f(b) have opposite signs, and |f(a)| >= |f(b)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = b, fb  # c is the previous b (smaller |f|)
    d, ed = a, a - b  # d is previous step

    for _ in range(60):
        if not _check_deadline():
            break
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a

        # Inverse quadratic interpolation
        mflag = True
        if c != a:
            # Try inverse quadratic interpolation
            r = (fb / fa) * ((b - c) / (fc - fb))
            s = (fb / fc) * ((b - a) / (fa - fb))
            t = 2.0 * min(0.0, r, s)
            u = t
            if abs(t) > abs(s):
                u = t
            elif abs(t) > abs(r):
                u = t
            # Standard Brent: use the formula
            # m = 0.5 * (c - b) * (fb/(fa-fb) - fa/(fc-fb)) + b
            denom1 = fa - fb
            denom2 = fc - fb
            if abs(denom1) > 1e-300 and abs(denom2) > 1e-300:
                m = 0.5 * (c - b) * (fb / denom1 - fa / denom2) + b
                # Safeguard conditions
                if ((m - (2.0 * a + b) / 3.0) * (m - b) > 0.0 or
                        (fb * m - b) * (fb * c - c) >= 0.0 or
                        abs(m - b) >= abs(b - c) / 2.0):
                    m = (a + b) / 2.0
                    d, ed = b - a, a - b
                else:
                    d = m - b
            else:
                m = (a + b) / 2.0
                d, ed = b - a, a - b

        if abs(d) < 1e-14 * max(1.0, abs(a), abs(b)):
            # Take bisection step
            b = (a + b) / 2.0
        else:
            b = b + d

        try:
            fb = f(b)
        except Exception:
            break
        if not math.isfinite(fb):
            break

        if abs(fb) <= 1e-8:
            return b

        if fa * fb > 0:
            # Same sign, replace a
            a, fa = b, fb
            c, fc = b, fb
            d, ed = b - a, a - b
        else:
            # Sign change, update bracket
            c, fc = a, fa
            d, ed = b - a, a - b
            a, fa = b, fb
            # Swap back so |fa| > |fb|
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa
            else:
                # Keep a,b as is but ensure c is the one with smaller |f|
                pass

        # Re-establish invariant: f(a) and f(b) opposite signs, |f(a)| >= |f(b)|
        if fa * fb > 0:
            # Shouldn't happen, but safeguard
            break
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

    # Return the endpoint with smaller |f|
    if abs(fa) <= abs(fb):
        return a
    else:
        return b


def _polish(f, x, fx):
    """High-precision Newton polish with central difference and tighter tolerance."""
    for _ in range(20):
        if not _check_deadline():
            break
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            break
        if abs(fx) <= 1e-12:
            break

        h = 1e-8 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except Exception:
            break
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            break

        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            break

        x_new = x - fx / d
        if not math.isfinite(x_new):
            break

        # Small damping for polish
        step = x_new - x
        if abs(step) > 0.1:
            x_new = x + (0.1 if step > 0 else -0.1)

        x = x_new
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break

    return x


def _bracket_and_solve(f, x_start, fx_start):
    """Try to find a bracket and solve using bisection/interpolation."""
    if not _check_deadline():
        return None
    if fx_start is None or not math.isfinite(fx_start):
        return None

    a = x_start
    fa = fx_start
    step = 1.0
    b = None
    fb = None

    # Expand to find a sign change
    for _ in range(30):
        if not _check_deadline():
            break
        b = a + step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        # If no sign change, try the other direction
        b = a - step
        try:
            fb = f(b)
        except Exception:
            step *= 2.0
            continue
        if not math.isfinite(fb):
            step *= 2.0
            continue
        if fa * fb <= 0:
            break
        step *= 2.0
    else:
        return None

    # Now we have a bracket [a, b] with fa * fb <= 0
    # Ensure a is the left endpoint
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))

    # Safeguarded inverse interpolation
    for _ in range(50):
        if not _check_deadline():
            break
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a

        denom = fb - fa
        if abs(denom) < 1e-300:
            x_new = (a + b) / 2.0
        else:
            x_new = (a * fb - b * fa) / denom

        # Safeguard: x_new must be in (a, b)
        if not (a < x_new < b):
            x_new = (a + b) / 2.0

        try:
            f_new = f(x_new)
        except Exception:
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
        if not math.isfinite(f_new):
            x_new = (a + b) / 2.0
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        if abs(f_new) < best_f:
            best_x = x_new
            best_f = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update bracket
        if fa * f_new <= 0:
            b = x_new
            fb = f_new
        else:
            a = x_new
            fa = f_new

        # Check convergence
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

    if abs(fb) <= abs(fa):
        return b
    else:
        return a
