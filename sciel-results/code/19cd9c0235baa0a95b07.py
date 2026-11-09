"""Direct root-finder: bracket-and-bisect plus Newton polish, with tight deadline guard."""
import time
import math

def find_root(f, x0):
    _t0 = time.time()
    _DEADLINE = 145.0  # safety margin before 150s
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        try:
            val = f(x)
            return float(val)
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _DEADLINE

    def _budget_ok():
        return _evals < _MAX_EVALS and _time_ok()

    # Best candidate
    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    def _bisection(a, b, fa, fb):
        """Bisect on [a,b] where fa*fb <= 0. Returns best point found."""
        nonlocal best_x, best_fx, best_abs
        # Ensure |fa| <= |fb| so we track the smaller one as 'a'
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        for _ in range(200):
            if not _budget_ok():
                break
            mid = (a + b) / 2.0
            if mid == a or mid == b:
                break
            fmid = _f(mid)
            _update(mid, fmid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        return mid if math.isfinite(mid) else (a + b) / 2.0

    def _secant_step(x, fx, x_prev, fx_prev):
        """Secant method step."""
        if fx_prev == fx:
            return None
        x_new = x - fx * (x - x_prev) / (fx - fx_prev)
        if not math.isfinite(x_new):
            return None
        return x_new

    def _newton_deriv(x, fx):
        """Compute derivative via central difference with adaptive h."""
        scale = max(1.0, abs(x))
        # Try a few h values
        for h_factor in [1e-5, 1e-4, 1e-3]:
            h = h_factor * scale
            if h == 0:
                continue
            fm = _f(x - h)
            fp = _f(x + h)
            if math.isfinite(fm) and math.isfinite(fp):
                d = (fp - fm) / (2.0 * h)
                if math.isfinite(d) and abs(d) > 1e-15:
                    return d
        # Forward difference fallback
        h = 1e-5 * scale
        fp = _f(x + h)
        if math.isfinite(fp) and math.isfinite(fx):
            d = (fp - fx) / h
            if math.isfinite(d) and abs(d) > 1e-15:
                return d
        return None

    def _newton_step(x, fx):
        d = _newton_deriv(x, fx)
        if d is None:
            return None
        step = fx / d
        # Dampen large steps
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Phase 1: Newton with bisection fallback ---
    x = float(x0)
    fx = _f(x)
    _update(x, fx)
    if math.isfinite(fx) and abs(fx) <= 1e-8:
        return x

    # Try Newton iterations, tracking a bracket when sign changes
    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None
    x_prev = x + 1e-6 * max(1.0, abs(x))
    fx_prev = _f(x_prev)
    _update(x_prev, fx_prev)

    for i in range(60):
        if not _budget_ok():
            break
        if abs(fx) <= 1e-8:
            return x
        # Try Newton
        x_new = _newton_step(x, fx)
        if x_new is None:
            # Try secant
            x_new = _secant_step(x, fx, x_prev, fx_prev)
        if x_new is None:
            break
        fx_new = _f(x_new)
        _update(x_new, fx_new)
        if not math.isfinite(fx_new):
            break
        if abs(fx_new) <= 1e-8:
            return x_new
        # Check for sign change to establish/refresh bracket
        if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
            if abs(fx) <= abs(fx_new):
                bracket_a, bracket_b = x, x_new
                bracket_fa, bracket_fb = fx, fx_new
            else:
                bracket_a, bracket_b = x_new, x
                bracket_fb, bracket_fa = fx, fx_new
        x_prev, fx_prev = x, fx
        x, fx = x_new, fx_new
        if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
            break

    if abs(fx) <= 1e-8 and math.isfinite(fx):
        return x
    _update(x, fx)

    # If we have a bracket, bisect
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            mid = _bisection(bracket_a, bracket_b, bracket_fa, bracket_fb)
            _update(mid, _f(mid))
            if abs(_f(mid)) <= 1e-8:
                return mid
            if math.isfinite(mid):
                _update(mid, _f(mid))

    # --- Phase 2: Scan for a bracket around x0 ---
    if _budget_ok() and best_abs > 1e-8:
        base = float(x0)
        scan_scale = max(1.0, abs(base))
        for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0]:
            if not _budget_ok():
                break
            for sign in [-1, 1]:
                if not _budget_ok():
                    break
                a = base - sign * mult * scan_scale
                b = base + sign * mult * scan_scale
                if a == b:
                    continue
                fa = _f(a)
                fb = _f(b)
                _update(a, fa)
                _update(b, fb)
                if not (math.isfinite(fa) and math.isfinite(fb)):
                    continue
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                    mid = _bisection(a, b, fa, fb)
                    fm = _f(mid)
                    _update(mid, fm)
                    if abs(fm) <= 1e-8:
                        return mid
                    if math.isfinite(mid):
                        return mid if abs(_f(mid)) <= best_abs else best_x
                    break
            else:
                continue
            break

    # --- Phase 3: Final Newton polish from best_x ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = _f(x)
        _update(x, fx)
        for _ in range(30):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-8:
                return x
            x_new = _newton_step(x, fx)
            if x_new is None:
                break
            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) > abs(fx):
                break
            x, fx = x_new, fx_new
            if abs(x - x_new) < 1e-15 * max(1.0, abs(x)):
                break

    # --- Phase 4: Secant polish (quadratic convergence, no derivative cost) ---
    if _budget_ok() and best_abs > 1e-8:
        x = best_x
        fx = _f(x)
        _update(x, fx)
        if not math.isfinite(fx):
            return best_x
        # Choose x_prev: try small perturbation to get a distinct point
        h = 1e-6 * max(1.0, abs(x))
        x_prev = x + h
        fx_prev = _f(x_prev)
        _update(x_prev, fx_prev)
        if not math.isfinite(fx_prev) or fx_prev == fx:
            # Try the other side
            x_prev = x - h
            fx_prev = _f(x_prev)
            _update(x_prev, fx_prev)
            if not math.isfinite(fx_prev) or fx_prev == fx:
                return best_x

        for _ in range(40):
            if not _budget_ok():
                break
            if abs(fx) <= 1e-8:
                return x
            if fx == fx_prev:
                break
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            fx_new = _f(x_new)
            _update(x_new, fx_new)
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) <= 1e-8:
                return x_new
            # Move the pair: keep the point with smaller |f| and the new one
            if abs(fx_new) < abs(fx):
                x_prev, fx_prev = x, fx
                x, fx = x_new, fx_new
            else:
                x_prev, fx_prev = x, fx
                # x_new is worse; use it as the "old" point for next secant
                x_prev, fx_prev = x_new, fx_new
            if abs(x - x_prev) < 1e-15 * max(1.0, abs(x)):
                break

    return best_x
