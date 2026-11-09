"""Root finder for 3 fixed oscillatory equations."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin well under 120s

    x = float(x0)

    # Check initial point
    fx = f(x)
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    # Try to find a bracket [a, b] with sign change
    a, b = None, None
    fa, fb = None, None

    # First try expanding from x0 in both directions geometrically
    step = 0.1
    for i in range(50):
        if time.time() > deadline:
            break
        # Try positive direction
        x_plus = x + step
        fx_plus = f(x_plus)
        if math.isfinite(fx_plus) and fx * fx_plus < 0:
            a, b = x, x_plus
            fa, fb = fx, fx_plus
            break
        # Try negative direction
        x_minus = x - step
        fx_minus = f(x_minus)
        if math.isfinite(fx_minus) and fx * fx_minus < 0:
            a, b = x_minus, x
            fa, fb = fx_minus, fx
            break
        step *= 2.0
        if step > 1e6:
            break

    # If no bracket found, try a wider search
    if a is None:
        # Try sampling at multiple points to find a sign change
        candidates = []
        base = x
        for scale in [1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            for sign in [-1, 1]:
                xp = base + sign * scale
                candidates.append(xp)
            if time.time() > deadline:
                break

        # Evaluate candidates and look for sign changes
        vals = []
        for xp in candidates:
            if time.time() > deadline:
                break
            try:
                fxp = f(xp)
                if math.isfinite(fxp):
                    vals.append((xp, fxp))
            except Exception:
                pass

        # Check for sign changes between consecutive points
        for i in range(len(vals) - 1):
            x1, f1 = vals[i]
            x2, f2 = vals[i + 1]
            if f1 * f2 < 0:
                a, b = min(x1, x2), max(x1, x2)
                fa, fb = f1, f2
                break

        # Also check against original x
        if a is None:
            for xp, fxp in vals:
                if fx * fxp < 0:
                    a, b = min(x, xp), max(x, xp)
                    fa, fb = fx, fxp
                    break

    # If still no bracket, use bisection on a wide interval or return best guess
    if a is None:
        # Last resort: try to minimize |f| via a few Newton-like steps
        x_best = x
        f_best = fx
        for i in range(20):
            if time.time() > deadline:
                break
            h = 1e-7 * max(1.0, abs(x_best))
            try:
                d = (f(x_best + h) - f_best) / h
            except Exception:
                break
            if d == 0 or not math.isfinite(d):
                break
            x_new = x_best - f_best / d
            if not math.isfinite(x_new):
                break
            x_best = x_new
            try:
                f_best = f(x_best)
            except Exception:
                break
            if not math.isfinite(f_best):
                break
            if abs(f_best) <= 1e-8:
                return x_best
        return x_best

    # Safeguarded Newton (Illinois method) within [a, b]
    # Ensure fa < fb in magnitude for Illinois damping
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    for i in range(100):
        if time.time() > deadline:
            break
        # Check if either endpoint is good enough
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b

        # Illinois method: bisection with damping on the side with smaller |f|
        # The side with larger |f| gets damped (multiplied by 0.5)
        if abs(fa) > abs(fb):
            # Damp fa side: new point closer to b
            m = 0.5 * (a + b)
            fm = f(m)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
                fa *= 0.5  # Illinois damping
        else:
            # Damp fb side: new point closer to a
            m = 0.5 * (a + b)
            fm = f(m)
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
                fb *= 0.5  # Illinois damping

        # Convergence check
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            break

    # Return the better of a or b
    if abs(fa) <= abs(fb):
        return a
    else:
        return b
# EVOLVE-BLOCK-END
