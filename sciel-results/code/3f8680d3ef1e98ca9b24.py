"""Combined root finder: Secant (1 call/iter) + Newton polish (central diff) + bracketing fallback.

Design:
- Phase 1: Secant method (1 f-call per iteration after initial setup).
- Phase 2: If Secant converges to ~1e-8, do a high-precision Newton polish with
  central difference (2 f-calls per iter) to tighten to ~1e-12, which guarantees
  passing the 1e-8 check with margin.
- Phase 3: If Secant fails, try bracketing + safeguarded inverse interpolation.
- Track best_x throughout; return best_x at the end.
"""
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

    # Get second point for Secant
    x1 = x + 1e-4 * max(1.0, abs(x))
    try:
        fx1 = f(x1)
    except Exception:
        if _check_deadline():
            x_bracketed = _bracket_and_solve(f, x, fx)
            if x_bracketed is not None:
                try:
                    fx_br = f(x_bracketed)
                    if math.isfinite(fx_br) and abs(fx_br) < abs(fx):
                        return x_bracketed
                except Exception:
                    pass
        return x

    if not math.isfinite(fx1):
        if _check_deadline():
            x_bracketed = _bracket_and_solve(f, x, fx)
            if x_bracketed is not None:
                try:
                    fx_br = f(x_bracketed)
                    if math.isfinite(fx_br) and abs(fx_br) < abs(fx):
                        return x_bracketed
                except Exception:
                    pass
        return x

    if abs(fx1) < abs(fx):
        best_x = x1
        best_fx = fx1

    # Phase 1: Secant Method
    a = x
    fa = fx
    b = x1
    fb = fx1

    converged = False
    for i in range(50):
        if not _check_deadline():
            break

        if abs(fa) <= 1e-8:
            best_x = a
            best_fx = fa
            converged = True
            break
        if abs(fb) <= 1e-8:
            best_x = b
            best_fx = fb
            converged = True
            break

        denom = fb - fa
        if abs(denom) < 1e-300:
            step = 1e-4 * max(1.0, abs(b))
            c = b + step
            try:
                fc = f(c)
            except Exception:
                break
            if not math.isfinite(fc):
                break
            a = b
            fa = fb
            b = c
            fb = fc
            continue

        c = b - fb * (b - a) / (fb - fa)

        # Damping: if step is huge, limit it
        if abs(c - b) > 10.0 * max(1.0, abs(b)):
            c = b + 10.0 * max(1.0, abs(b)) * (1 if c > b else -1)

        if not math.isfinite(c):
            break

        try:
            fc = f(c)
        except Exception:
            break
        if not math.isfinite(fc):
            break

        if abs(fc) < abs(best_fx):
            best_x = c
            best_fx = fc

        a = b
        fa = fb
        b = c
        fb = fc

        if abs(fc) <= 1e-8:
            converged = True
            break

        # If oscillating or stalled
        if i > 20 and abs(b - a) < 1e-12 * max(1.0, abs(b)):
            break

    # Phase 2: High-precision Newton polish (central difference)
    # This uses 2 calls per iteration but converges quadratically, so very few
    # iterations needed. We polish from the best point found.
    if best_fx is not None and abs(best_fx) <= 1e-5 and _check_deadline():
        x_polished = _polish(f, best_x, best_fx)
        if x_polished is not None:
            try:
                fx_p = f(x_polished)
                if math.isfinite(fx_p) and abs(fx_p) < abs(best_fx):
                    best_x = x_polished
                    best_fx = fx_p
            except Exception:
                pass

    # Phase 3: If still not converged, try bracketing
    if best_fx is not None and abs(best_fx) > 1e-8 and _check_deadline():
        x_bracketed = _bracket_and_solve(f, best_x, best_fx)
        if x_bracketed is not None:
            try:
                fx_br = f(x_bracketed)
                if math.isfinite(fx_br) and abs(fx_br) < abs(best_fx):
                    best_x = x_bracketed
                    best_fx = fx_br
            except Exception:
                pass

        # One more polish attempt if bracketing got us close
        if best_fx is not None and abs(best_fx) <= 1e-5 and _check_deadline():
            x_polished = _polish(f, best_x, best_fx)
            if x_polished is not None:
                try:
                    fx_p = f(x_polished)
                    if math.isfinite(fx_p) and abs(fx_p) < abs(best_fx):
                        best_x = x_polished
                        best_fx = fx_p
                except Exception:
                    pass

    return best_x


def _polish(f, x, fx):
    """High-precision Newton polish with central difference. Returns None on failure."""
    if fx is None or not math.isfinite(fx):
        return None
    if abs(fx) <= 1e-12:
        return x

    for _ in range(30):
        if not _check_deadline():
            return x
        if not math.isfinite(x) or fx is None or not math.isfinite(fx):
            return x
        if abs(fx) <= 1e-12:
            return x

        h = 1e-8 * max(1.0, abs(x))
        try:
            f_xp = f(x + h)
            f_xm = f(x - h)
        except Exception:
            return x
        if not math.isfinite(f_xp) or not math.isfinite(f_xm):
            return x

        d = (f_xp - f_xm) / (2.0 * h)
        if abs(d) < 1e-16:
            return x

        x_new = x - fx / d
        if not math.isfinite(x_new):
            return x

        # Damping for safety
        step = x_new - x
        if abs(step) > 0.5 * max(1.0, abs(x)):
            x_new = x + 0.5 * max(1.0, abs(x)) * (1 if step > 0 else -1)

        x = x_new
        try:
            fx = f(x)
        except Exception:
            return x
        if not math.isfinite(fx):
            return x

    return x


def _bracket_and_solve(f, x_start, fx_start):
    """Find a bracket and solve using safeguarded inverse interpolation."""
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
    found_bracket = False
    for _ in range(30):
        if not _check_deadline():
            break
        # Try right
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
            found_bracket = True
            break
        # Try left
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
            found_bracket = True
            break
        step *= 2.0

    if not found_bracket:
        return None

    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    best_x = a if abs(fa) < abs(fb) else b
    best_f = min(abs(fa), abs(fb))

    # Safeguarded inverse interpolation (1 call per iter)
    for _ in range(60):
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
