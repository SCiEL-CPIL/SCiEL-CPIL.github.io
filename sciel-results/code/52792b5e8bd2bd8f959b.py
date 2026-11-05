import time
import math
import numpy as np

_DEADLINE = 150.0
_MARGIN = 25.0
_EVAL_BUDGET = 490
_ABS_TOL = 1e-8


class _Budget:
    __slots__ = ("count", "start", "ok")

    def __init__(self):
        self.count = 0
        self.start = time.time()
        self.ok = True

    def eval(self, f, x):
        if not self.ok:
            return 0.0
        self.count += 1
        if self.count > _EVAL_BUDGET or (time.time() - self.start) > (_DEADLINE - _MARGIN):
            self.ok = False
            return 0.0
        try:
            v = f(float(x))
        except Exception:
            self.ok = False
            return 0.0
        if not math.isfinite(v):
            return float("inf")
        return v


def _iqi_step(x0, f0, x1, f1, x2, f2):
    """Inverse quadratic interpolation step. Returns x_new or None if not applicable."""
    # Conditions for IQI to be well-defined
    if f0 == f1 or f0 == f2 or f1 == f2:
        return None
    # IQI formula
    L0 = f0 * (f0 - f1) * (f0 - f2)
    L1 = f1 * (f1 - f0) * (f1 - f2)
    L2 = f2 * (f2 - f0) * (f2 - f1)
    denom = L0 + L1 + L2
    if abs(denom) < 1e-30:
        return None
    x_new = (x0 * L0 + x1 * L1 + x2 * L2) / denom
    # Sanity check: should be close to one of the points
    scale = max(abs(x0), abs(x1), abs(x2), 1.0)
    if not (min(x0, x1, x2) - 2 * scale < x_new < max(x0, x1, x2) + 2 * scale):
        return None
    return x_new


def _secant_with_iqi(f, x0, fx0, x1, fx1, B):
    """
    Robust root finding using secant + IQI with safeguards.
    Maintains three points (x0,f0), (x1,f1), (x2,f2) and uses IQI when possible,
    falling back to secant. Safeguards against divergence.
    Returns (x_best, f_best).
    """
    # State: three most recent points
    xa, fa = x0, fx0
    xb, fb = x1, fx1
    # Need a third point; use secant first
    if fa == fb:
        # Degenerate: try perturbation
        h = 1e-6 * max(1.0, abs(xa))
        xc = xa + h
        fc = B.eval(f, xc)
        if fc == float("inf"):
            xc = xa - h
            fc = B.eval(f, xc)
            if fc == float("inf"):
                return xa if abs(fa) <= abs(fb) else xb, min(abs(fa), abs(fb))
        # Now have three points
    else:
        # Secant step to get third point
        slope = (fb - fa) / (xb - xa)
        if slope == 0:
            return xa if abs(fa) <= abs(fb) else xb, min(abs(fa), abs(fb))
        xc = xb - fb / slope
        fc = B.eval(f, xc)
        if fc == float("inf"):
            # Bad step, try bisection-like
            xc = 0.5 * (xa + xb)
            fc = B.eval(f, xc)
            if fc == float("inf"):
                return xa if abs(fa) <= abs(fb) else xb, min(abs(fa), abs(fb))

    # Now we have three points: (xa,fa), (xb,fb), (xc,fc)
    # Track best
    best_x = xa
    best_f = abs(fa)
    if abs(fb) < best_f:
        best_x, best_f = xb, abs(fb)
    if abs(fc) < best_f:
        best_x, best_f = xc, abs(fc)

    if best_f <= _ABS_TOL:
        return best_x, best_f

    # Main iteration: use IQI or secant
    x0, f0 = xa, fa
    x1, f1 = xb, fb
    x2, f2 = xc, fc

    for _ in range(150):
        if not B.ok:
            break

        # Update best
        if abs(f2) < best_f:
            best_f = abs(f2)
            best_x = x2
        if abs(f1) < best_f:
            best_f = abs(f1)
            best_x = x1
        if abs(f0) < best_f:
            best_f = abs(f0)
            best_x = x0

        if best_f <= _ABS_TOL:
            return best_x, best_f

        # Try IQI first
        x_new = _iqi_step(x0, f0, x1, f1, x2, f2)

        if x_new is None:
            # Fall back to secant on the two most recent points with smallest |f|
            # Choose pair: prefer points that give a reasonable step
            # Use (x1, f1) and (x2, f2) as the most recent pair
            if f1 == f2:
                # Degenerate, perturb
                h = 1e-6 * max(1.0, abs(x2))
                x_new = x2 + h
                f_new = B.eval(f, x_new)
                if f_new == float("inf"):
                    x_new = x2 - h
                    f_new = B.eval(f, x_new)
                    if f_new == float("inf"):
                        break
            else:
                slope = (f2 - f1) / (x2 - x1)
                if slope == 0:
                    break
                x_new = x2 - f2 / slope
                f_new = B.eval(f, x_new)
                if f_new == float("inf"):
                    # Bad step, use bisection between x1 and x2
                    x_new = 0.5 * (x1 + x2)
                    f_new = B.eval(f, x_new)
                    if f_new == float("inf"):
                        break
        else:
            f_new = B.eval(f, x_new)
            if f_new == float("inf"):
                # IQI gave bad point, fall back to secant
                if f1 == f2:
                    h = 1e-6 * max(1.0, abs(x2))
                    x_new = x2 + h
                    f_new = B.eval(f, x_new)
                    if f_new == float("inf"):
                        x_new = x2 - h
                        f_new = B.eval(f, x_new)
                        if f_new == float("inf"):
                            break
                else:
                    slope = (f2 - f1) / (x2 - x1)
                    if slope == 0:
                        break
                    x_new = x2 - f2 / slope
                    f_new = B.eval(f, x_new)
                    if f_new == float("inf"):
                        x_new = 0.5 * (x1 + x2)
                        f_new = B.eval(f, x_new)
                        if f_new == float("inf"):
                            break

        # Safeguard: if step is too large, dampen it
        scale = max(abs(x2), abs(x_new), 1.0)
        if abs(x_new - x2) > 10.0 * scale:
            x_new = x2 + 0.5 * (x_new - x2)
            f_new = B.eval(f, x_new)
            if f_new == float("inf"):
                x_new = 0.5 * (x1 + x2)
                f_new = B.eval(f, x_new)
                if f_new == float("inf"):
                    break

        # Update triple: shift (x0,f0) -> (x1,f1), (x1,f1) -> (x2,f2), add new point
        x0, f0 = x1, f1
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new

        # Convergence check on step
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
            break

    # Final best
    if abs(f2) < best_f:
        best_f = abs(f2)
        best_x = x2
    if abs(f1) < best_f:
        best_f = abs(f1)
        best_x = x1
    if abs(f0) < best_f:
        best_f = abs(f0)
        best_x = x0

    return best_x, best_f


def find_root(f, x0):
    B = _Budget()
    x0 = float(x0)
    fx0 = B.eval(f, x0)
    if not B.ok:
        return x0
    if abs(fx0) <= _ABS_TOL:
        return x0

    best_x, best_f = x0, fx0

    # Multi-start: try several perturbed starting points
    # The perturbations are chosen to be small enough to stay in the basin
    # but large enough to escape flat regions or singularities
    base_scale = max(1.0, abs(x0))
    # Perturbation magnitudes: geometrically spaced
    perturbations = [
        1e-6 * base_scale,
        1e-4 * base_scale,
        1e-2 * base_scale,
        1e-1 * base_scale,
        1.0 * base_scale,
    ]

    for h in perturbations:
        if not B.ok:
            break
        # Try both directions
        for sign in (1.0, -1.0):
            if not B.ok:
                break
            x_start = x0 + sign * h
            fx_start = B.eval(f, x_start)
            if fx_start == float("inf"):
                continue
            if abs(fx_start) <= _ABS_TOL:
                return x_start
            if abs(fx_start) < abs(best_f):
                best_x, best_f = x_start, fx_start

            # Need a second point for secant/IQI
            # Use a small perturbation from x_start
            h2 = 1e-5 * max(1.0, abs(x_start))
            x_second = x_start + h2
            fx_second = B.eval(f, x_second)
            if fx_second == float("inf"):
                x_second = x_start - h2
                fx_second = B.eval(f, x_second)
                if fx_second == float("inf"):
                    continue

            # Run the IQI+secant method
            try:
                xr, fxr = _secant_with_iqi(f, x_start, fx_start, x_second, fx_second, B)
                if abs(fxr) < abs(best_f):
                    best_x, best_f = xr, fxr
                if abs(fxr) <= _ABS_TOL:
                    return xr
            except Exception:
                pass

        # Early exit if we've found a good solution
        if abs(best_f) <= _ABS_TOL:
            return best_x

    # Final attempt: pure secant from original x0 with a reasonable step
    if B.ok:
        try:
            h = 1e-3 * base_scale
            x1 = x0 + h
            fx1 = B.eval(f, x1)
            if fx1 != float("inf"):
                xr, fxr = _secant_with_iqi(f, x0, fx0, x1, fx1, B)
                if abs(fxr) < abs(best_f):
                    best_x, best_f = xr, fxr
        except Exception:
            pass

    return best_x
