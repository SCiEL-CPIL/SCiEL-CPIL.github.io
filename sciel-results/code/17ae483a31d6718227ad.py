"""Root finder for oscillatory equations: inverse quadratic interpolation primary, then bracketing."""
import math
import time

# Evaluator kills at 150s wall-clock; use 140s deadline for safety.
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480  # Stay under 500 with margin


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

    x0f = float(x0)
    base = max(1.0, abs(x0f))

    # --- Initial evaluation at x0 ---
    fx0, ok0 = safe_eval(x0f)
    if ok0 and try_update(x0f, fx0):
        return best_x

    # Cache to avoid re-evaluation
    cache = {}
    if ok0:
        cache[x0f] = (fx0, True)

    # --- Phase 1: Inverse Quadratic Interpolation (IQI) from x0 ---
    # IQI uses 3 points to model x as a quadratic in f, achieving superlinear
    # convergence with only 1 new function eval per step after the initial 3.
    if ok0 and counter[0] < max_calls and _time_left() > 5.0:
        # Get two more points to start IQI
        h1 = 1e-4 * base
        h2 = 2e-4 * base
        x1 = x0f + h1
        x2 = x0f - h2
        fx1, ok1 = safe_eval(x1)
        if ok1 and try_update(x1, fx1):
            return best_x
        fx2, ok2 = safe_eval(x2)
        if ok2 and try_update(x2, fx2):
            return best_x

        if ok1 and ok2:
            # We have three points: (x0, fx0), (x1, fx1), (x2, fx2)
            # Perform IQI iterations
            pts = [(x0f, fx0), (x1, fx1), (x2, fx2)]
            for _ in range(30):
                if counter[0] >= max_calls or _time_left() < 5.0:
                    break

                x_a, f_a = pts[0]
                x_b, f_b = pts[1]
                x_c, f_c = pts[2]

                # Check if any is a root
                for xp, fp in pts:
                    if abs(fp) <= 1e-8:
                        return xp

                # Inverse quadratic interpolation:
                # x = x_a * L_a(f) + x_b * L_b(f) + x_c * L_c(f)
                # where L_a(f) = (f - f_b)(f - f_c) / ((f_a - f_b)(f_a - f_c))
                # and we want x such that f(x) = 0
                denom_a = (f_a - f_b) * (f_a - f_c)
                denom_b = (f_b - f_a) * (f_b - f_c)
                denom_c = (f_c - f_a) * (f_c - f_b)

                if abs(denom_a) < 1e-300 or abs(denom_b) < 1e-300 or abs(denom_c) < 1e-300:
                    # Fallback to secant
                    if abs(f_b - f_c) < 1e-300:
                        break
                    x_new = x_b - f_b * (x_b - x_c) / (f_b - f_c)
                else:
                    # Evaluate at f=0
                    num_a = (0 - f_b) * (0 - f_c)
                    num_b = (0 - f_a) * (0 - f_c)
                    num_c = (0 - f_a) * (0 - f_b)
                    x_new = x_a * num_a / denom_a + x_b * num_b / denom_b + x_c * num_c / denom_c

                if not math.isfinite(x_new):
                    break

                # Damp the step
                # Find the point closest to x_new to measure step
                ref_x = min(pts, key=lambda p: abs(p[0] - x_new))[0]
                step = x_new - ref_x
                max_step = 2.0 * max(1.0, abs(ref_x))
                if abs(step) > max_step:
                    x_new = ref_x + max_step * math.copysign(1.0, step)

                fx_new, ok_new = safe_eval(x_new)
                if not ok_new:
                    break
                if try_update(x_new, fx_new):
                    return best_x

                # Update points: keep the two closest to x_new and the new point
                # Sort by distance to x_new
                all_pts = pts + [(x_new, fx_new)]
                all_pts.sort(key=lambda p: abs(p[0] - x_new))
                pts = all_pts[:3]

    # --- Phase 2: Secant method from best_x ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 5.0:
        x = float(best_x)
        fx, okx = safe_eval(x)
        if not okx:
            fx = fx0 if ok0 else float('nan')
            okx = ok0
        if okx and try_update(x, fx):
            return best_x

        h = 1e-4 * max(1.0, abs(x))
        x1 = x + h
        fx1, ok1 = safe_eval(x1)
        if ok1 and try_update(x1, fx1):
            return best_x

        if okx and ok1:
            for _ in range(20):
                if counter[0] >= max_calls or _time_left() < 5.0:
                    break
                if abs(fx) <= 1e-8:
                    return x
                if abs(fx1) <= 1e-8:
                    return x1
                denom = fx1 - fx
                if abs(denom) < 1e-300:
                    break
                x_new = x1 - fx1 * (x1 - x) / denom
                if not math.isfinite(x_new):
                    break
                step = x_new - x1
                max_step = 2.0 * max(1.0, abs(x1))
                if abs(step) > max_step:
                    x_new = x1 + max_step * math.copysign(1.0, step)
                fx_new, ok_new = safe_eval(x_new)
                if not ok_new:
                    break
                if try_update(x_new, fx_new):
                    return best_x
                x, fx = x1, fx1
                x1, fx1 = x_new, fx_new

    # --- Phase 3: Multi-scale dense sampling to find a bracket ---
    bracket_found = False
    a_br, b_br = None, None
    fa_br, fb_br = None, None

    scales = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 200.0, 500.0]
    n_samples = [7, 9, 11, 13, 15, 17, 19, 21, 25, 30, 35, 40, 45, 50, 55, 60, 65]

    for idx, scale in enumerate(scales):
        if counter[0] >= max_calls or _time_left() < 3.0:
            break
        h = scale * base
        n = n_samples[idx]
        half = n // 2
        points = []
        for i in range(-half, half + 1):
            p = x0f + i * h
            if p not in cache:
                val, ok = safe_eval(p)
                cache[p] = (val, ok)
                if ok and try_update(p, val):
                    return best_x
            points.append(p)

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

    # --- Phase 4: Brent's method on the bracket ---
    if bracket_found:
        try:
            res = _brentq(wf, a_br, b_br, xtol=1e-14, rtol=1e-14, maxiter=50)
            if math.isfinite(res):
                fr, okr = safe_eval(res)
                if okr and try_update(res, fr):
                    return best_x
                if okr and abs(fr) < best_abs:
                    best_x = res
                    best_abs = abs(fr)
        except Exception:
            pass

    # --- Phase 5: Fine local scan around best_x ---
    if best_abs > 1e-8 and counter[0] < max_calls and _time_left() > 2.0:
        for scale in [1e-5, 1e-6, 1e-7, 1e-8]:
            if counter[0] >= max_calls or _time_left() < 2.0:
                break
            h = scale * max(1.0, abs(best_x))
            for i in range(-5, 6):
                if counter[0] >= max_calls:
                    break
                p = best_x + i * h
                if p not in cache:
                    val, ok = safe_eval(p)
                    cache[p] = (val, ok)
                    if ok and try_update(p, val):
                        return best_x

    return best_x


def _brentq(f, a, b, xtol=1e-14, rtol=1e-14, maxiter=50):
    """Brent's method implementation."""
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("f(a) and f(b) must have different signs")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    c = a
    fc = fa
    d = b - a
    e = d
    for _ in range(maxiter):
        if abs(b - a) < xtol + rtol * abs(b):
            return b
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - c
            e = d
        if abs(fc) < abs(fb):
            c = b
            fc = fb
            a = b
            fa = fb
            d = b - c
            e = d
        # Inverse quadratic interpolation
        tol = 2e-16 * max(1.0, abs(b))
        tm = 0.5 * (c - b)
        if abs(tm) <= tol or fc == fb:
            # Bisection
            s = fc / (fc - fb)
            e = d
            d = tm * s
        else:
            r = fc / fa
            q = fb / fa
            s = fb / fc
            if c == a:
                d = tm * (2 * tm * (fb / (fc - fa) - 1) + s * (s - 1))
            else:
                d = tm * (2 * tm * (fb / (fc - fa) - 1) + s * (s - 1))
            if abs(d) >= abs(e):
                d = tm
                e = d
            else:
                e = d
        if d > 0:
            d = -d
        a = b
        fa = fb
        b += d
        fb = f(b)
    return b
