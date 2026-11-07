"""Scalar root finder for 3 fixed smooth polynomial equations.

Strategy: Halley's method (3 f-evals/step) with bisection safety,
adaptive finite-difference derivatives, a wall-clock deadline guard,
and a final high-precision polish pass.
"""
# EVOLVE-BLOCK-START
import math
import time

_DEADLINE = 110.0  # safety margin before 120 s / 150 s kill
_TOL = 1e-13       # internal tolerance (well under 1e-8)
_MIN_TOL = 1e-16


def _now():
    return time.monotonic()


def _fd1(f, x, fx, h):
    return (f(x + h) - fx) / h


def _fd2(f, x, fx, h):
    return (f(x + h) - 2.0 * fx + f(x - h)) / (h * h)


def _safe_bracket(f, x0, fx0):
    """Find a bracket [a, b] with opposite signs around x0."""
    a, b = x0, x0
    fa = fx0
    step = 1.0
    # expand outward
    for _ in range(80):
        if abs(fa) <= _TOL:
            return (x0, x0)  # already a root
        # try right
        xr = x0 + step
        fr = f(xr)
        if not math.isfinite(fr):
            step *= 2.0
            continue
        if fa * fr < 0:
            return (x0, xr)
        # try left
        xl = x0 - step
        fl = f(xl)
        if not math.isfinite(fl):
            step *= 2.0
            continue
        if fa * fl < 0:
            return (xl, x0)
        step *= 2.0
        if step > 1e12:
            break
    return None


def _bisect(f, a, b, fa, fb, deadline):
    """Bisection until |f| <= _TOL or deadline. Returns (x, fx, converged)."""
    for _ in range(200):
        if _now() > deadline:
            mid = 0.5 * (a + b)
            fm = f(mid)
            if abs(fm) < abs(fa):
                return mid, fm, False
            return a, fa, False
        mid = 0.5 * (a + b)
        fm = f(mid)
        if abs(fm) <= _TOL:
            return mid, fm, True
        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm
        if b - a <= _MIN_TOL * max(1.0, abs(mid)):
            return mid, fm, abs(fm) <= _TOL
    mid = 0.5 * (a + b)
    fm = f(mid)
    return mid, fm, abs(fm) <= _TOL


def _halley_step(f, x, fx, h):
    """One Halley step. Returns (new_x, converged, safe)."""
    f1 = _fd1(f, x, fx, h)
    if f1 == 0.0 or not math.isfinite(f1):
        return x, False, False
    f2 = _fd2(f, x, fx, h)
    # Halley's method: x - 2 f f' / (2 f'^2 - f f'')
    denom = 2.0 * f1 * f1 - fx * f2
    if abs(denom) < 1e-300:
        # fall back to Newton
        nx = x - fx / f1
        if not math.isfinite(nx) or abs(nx - x) > 1e10 * max(1.0, abs(x)):
            return x, False, False
        return nx, False, True
    nx = x - 2.0 * fx * f1 / denom
    if not math.isfinite(nx):
        return x, False, False
    # reject huge jumps
    if abs(nx - x) > 1e10 * max(1.0, abs(x)):
        return x, False, False
    return nx, False, True


def _polish(f, x, fx, deadline):
    """High-precision Newton polish with machine-eps step."""
    for _ in range(20):
        if _now() > deadline:
            break
        if abs(fx) <= _MIN_TOL:
            break
        h = 1e-11 * max(1.0, abs(x))
        f1 = _fd1(f, x, fx, h)
        if f1 == 0.0 or not math.isfinite(f1):
            break
        dx = fx / f1
        x_new = x - dx
        if not math.isfinite(x_new):
            break
        fx_new = f(x_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) >= abs(fx):
            break
        x, fx = x_new, fx_new
        if abs(dx) <= 1e-15 * max(1.0, abs(x)):
            break
    return x, fx


def find_root(f, x0):
    deadline = _now() + _DEADLINE
    x = float(x0)
    fx = f(x)

    # Already a root?
    if math.isfinite(fx) and abs(fx) <= _TOL:
        return x

    # Best-so-far tracking
    best_x, best_fx = x, fx
    best_abs = abs(fx) if math.isfinite(fx) else float('inf')

    def _update(xc, fxc):
        nonlocal best_x, best_fx, best_abs
        if math.isfinite(fxc) and abs(fxc) < best_abs:
            best_x, best_fx, best_abs = xc, fxc, abs(fxc)
            if best_abs <= _TOL:
                return True
        return False

    # Try to get a bracket for safety
    bracket = _safe_bracket(f, x, fx)
    a = b = None
    if bracket is not None:
        a, b = bracket
        if a == b:
            return a  # exact root found during bracketing

    # Adaptive h
    h = 1e-6 * max(1.0, abs(x))

    for it in range(60):
        if _now() > deadline:
            break
        if abs(fx) <= _TOL:
            break

        # Halley step
        nx, conv, safe = _halley_step(f, x, fx, h)
        if not safe:
            # Fall back to bisection if we have a bracket
            if a is not None and b is not None and a != b:
                x, fx, done = _bisect(f, a, b, f(a), f(b), deadline)
                _update(x, fx)
                if done or _now() > deadline:
                    break
                continue
            else:
                # No bracket; try Newton
                f1 = _fd1(f, x, fx, h)
                if f1 == 0.0 or not math.isfinite(f1):
                    break
                nx = x - fx / f1
                if not math.isfinite(nx) or abs(nx - x) > 1e10 * max(1.0, abs(x)):
                    break

        # Check if step is within bracket (if we have one)
        if a is not None and b is not None and a != b:
            lo, hi = min(a, b), max(a, b)
            if nx < lo or nx > hi:
                # Step left bracket; use bisection instead
                mid = 0.5 * (a + b)
                fm = f(mid)
                if not math.isfinite(fm):
                    break
                _update(mid, fm)
                if abs(fm) <= _TOL:
                    x, fx = mid, fm
                    break
                if f(a) * fm < 0:
                    b = mid
                else:
                    a = mid
                x, fx = mid, fm
                continue

        nx_fx = f(nx)
        if not math.isfinite(nx_fx):
            # Step went bad; bisect if possible
            if a is not None and b is not None and a != b:
                x, fx, done = _bisect(f, a, b, f(a), f(b), deadline)
                _update(x, fx)
                if done or _now() > deadline:
                    break
                continue
            break

        _update(nx, nx_fx)

        # Update bracket if we have one
        if a is not None and b is not None and a != b:
            fa = f(a)
            if fa * nx_fx < 0:
                b = nx
            else:
                a = nx

        x, fx = nx, nx_fx

        # Shrink h adaptively
        h = max(1e-11, h * 0.7)

    # Final polish
    x, fx = _polish(f, x, fx, deadline)
    _update(x, fx)

    # Return best point
    if _now() > deadline:
        return best_x

    # Final check: if best is good enough, return it
    if best_abs <= 1e-8:
        return best_x

    # If we have a bracket, do a final bisection burst
    if a is not None and b is not None and a != b:
        x, fx, done = _bisect(f, a, b, f(a), f(b), deadline)
        _update(x, fx)

    return best_x
# EVOLVE-BLOCK-END
