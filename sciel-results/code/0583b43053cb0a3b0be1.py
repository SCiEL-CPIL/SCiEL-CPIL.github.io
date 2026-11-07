"""Efficient root finder for oscillatory equations, minimizing call count."""
import math
import time
from scipy.optimize import brentq

# Evaluator kills at 150s wall-clock; use 140s deadline for safety.
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 450  # Stay comfortably under 500


def _time_left():
    return _DEADLINE - time.time()


def find_root(f, x0):
    """Find x such that |f(x)| <= 1e-8 for an oscillatory function f."""
    counter = [0]
    max_calls = _CALL_BUDGET

    def wf(x):
        counter[0] += 1
        if counter[0] > max_calls:
            raise RuntimeError("call budget exceeded")
        return f(x)

    best_x = float(x0)
    best_abs = float('inf')

    def try_update(x, val):
        nonlocal best_x, best_abs
        if math.isfinite(val):
            a = abs(val)
            if a < best_abs:
                best_x = x
                best_abs = a
                if best_abs <= 1e-8:
                    return True
        return False

    def safe_eval(x):
        """Evaluate wf(x), return (value, ok). ok=False if exception/non-finite."""
        try:
            v = wf(x)
            if not math.isfinite(v):
                return v, False
            return v, True
        except Exception:
            return float('nan'), False

    # --- Initial evaluation at x0 ---
    fx0, ok0 = safe_eval(float(x0))
    if ok0 and try_update(float(x0), fx0):
        return best_x

    # --- Strategy 1: Bracketing via geometric scale ladder ---
    # For oscillatory functions, roots are often nearby. Try scales from small to large.
    # At each scale, evaluate a few candidate points and look for sign change or near-zero.
    # Stop as soon as a bracket is found.

    bracket_found = False
    a_br, b_br = None, None
    fa_br, fb_br = None, None

    # Pre-compute base for scaling
    base = max(1.0, abs(float(x0)))
    x0f = float(x0)

    # Geometric ladder of half-widths. Start small to find nearby roots cheaply.
    scales = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]

    # Cache of evaluated points to avoid re-evaluation: dict x -> (val, ok)
    cache = {}
    if ok0:
        cache[x0f] = (fx0, True)

    for scale in scales:
        if counter[0] >= max_calls or _time_left() < 5.0:
            break
        h = scale * base
        # Candidate offsets: symmetric around x0, plus x0 itself
        # Use a small set: x0, x0±h, x0±2h, x0±3h
        offsets = [0.0, h, -h, 2*h, -2*h, 3*h, -3*h]
        points = []
        for off in offsets:
            p = x0f + off
            if p not in cache:
                val, ok = safe_eval(p)
                cache[p] = (val, ok)
                if ok and try_update(p, val):
                    return best_x
            points.append(p)

        # Check consecutive pairs for sign change
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            v1, ok1 = cache[p1]
            v2, ok2 = cache[p2]
            if not ok1 or not ok2:
                continue
            if v1 == 0.0:
                return p1
            if v2 == 0.0:
                return p2
            if v1 * v2 < 0.0:
                a_br, b_br = p1, p2
                fa_br, fb_br = v1, v2
                bracket_found = True
                break
        if bracket_found:
            break

    # --- Strategy 2: brentq on the bracket ---
    if bracket_found:
        try:
            res = brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=60)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Strategy 3: Newton with damped steps if no bracket or not converged ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 3.0:
        x = float(best_x)
        for _ in range(30):
            if counter[0] >= max_calls or _time_left() < 3.0:
                break
            fx, okx = safe_eval(x)
            if not okx:
                break
            if try_update(x, fx):
                return best_x
            # Numerical derivative via central difference
            h = 1e-7 * max(1.0, abs(x))
            fxp, okp = safe_eval(x + h)
            if not okp:
                break
            fxm, okm = safe_eval(x - h)
            if not okm:
                break
            d = (fxp - fxm) / (2.0 * h)
            if d == 0.0 or not math.isfinite(d):
                # Fallback to forward difference
                d = (fxp - fx) / h
                if d == 0.0 or not math.isfinite(d):
                    break
            x_new = x - fx / d
            if not math.isfinite(x_new):
                break
            # Damp the step
            step = x_new - x
            max_step = 5.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                x_new = x + max_step * math.copysign(1.0, step)
            x = x_new

    # --- Strategy 4: Fine local scan around best_x ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 2.0:
        for scale in [1e-5, 1e-6, 1e-7, 1e-8]:
            if counter[0] >= max_calls or _time_left() < 2.0:
                break
            h = scale * max(1.0, abs(best_x))
            for i in range(-4, 5):
                if counter[0] >= max_calls:
                    break
                p = best_x + i * h
                val, ok = safe_eval(p)
                if ok and try_update(p, val):
                    return best_x

    return best_x
