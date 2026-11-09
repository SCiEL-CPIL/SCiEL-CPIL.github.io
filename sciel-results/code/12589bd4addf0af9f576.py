"""Robust ill-conditioned root finder using model-and-optimize strategy."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 135.0  # safety margin before 150s
    _MAX_EVALS = 490
    _evals = 0

    def _f(x):
        nonlocal _evals
        _evals += 1
        return f(x)

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _safe_f(x):
        try:
            val = float(_f(x))
            return val
        except Exception:
            return float('nan')

    # Best candidate tracking
    best_x = float(x0)
    best_fx = _safe_f(best_x)
    best_abs = abs(best_fx) if math.isfinite(best_fx) else float('inf')

    def _update_best(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not math.isfinite(fx):
            return
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)

    _update_best(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    # --- Finite-difference derivative (central preferred, forward fallback) ---
    def _deriv(x, fx):
        scale = max(1.0, abs(x))
        # Try central difference first (more accurate)
        for h_mult in [1e-5, 1e-4, 1e-3]:
            h = h_mult * scale
            fp = _safe_f(x + h)
            fm = _safe_f(x - h)
            if math.isfinite(fp) and math.isfinite(fm) and math.isfinite(fx):
                d = (fp - fm) / (2.0 * h)
                if math.isfinite(d) and abs(d) > 1e-30:
                    return d
        # Fallback: forward difference
        for h_mult in [1e-5, 1e-4, 1e-3]:
            h = h_mult * scale
            fp = _safe_f(x + h)
            if math.isfinite(fp) and math.isfinite(fx):
                d = (fp - fx) / h
                if math.isfinite(d) and abs(d) > 1e-30:
                    return d
        return 0.0

    # --- Newton step with safeguard ---
    def _newton_step(x, fx):
        d = _deriv(x, fx)
        if d == 0 or not math.isfinite(d):
            return None
        step = fx / d
        # Limit step size to avoid wild jumps
        max_step = 5.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Inverse Quadratic Interpolation step ---
    def _iqi_step(x1, fx1, x2, fx2, x3, fx3):
        """IQI: given 3 points, find next iterate via inverse quadratic interpolation."""
        # All three must be finite and distinct
        if not all(math.isfinite(v) for v in [x1, fx1, x2, fx2, x3, fx3]):
            return None
        if x1 == x2 or x2 == x3 or x1 == x3:
            return None
        
        # Lagrange interpolation for inverse: x = p(f)
        # Using the standard IQI formula
        d1 = (fx1 - fx2) * (fx1 - fx3)
        d2 = (fx2 - fx1) * (fx2 - fx3)
        d3 = (fx3 - fx1) * (fx3 - fx2)
        
        denom = d1 + d2 + d3
        if abs(denom) < 1e-30:
            return None
        
        x_new = (x1 * d1 + x2 * d2 + x3 * d3) / denom
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Bisection on bracket [a, b] where f(a)*f(b) <= 0 ---
    def _bisection(a, b, fa, fb, max_iter=100):
        if not _time_ok():
            return None
        if not ((fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0)):
            return None
        for _ in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            mid = (a + b) / 2.0
            fmid = _safe_f(mid)
            if not math.isfinite(fmid):
                break
            _update_best(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-15 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        return mid

    # --- Robust hybrid solver: IQI + Newton + Bisection ---
    def _hybrid_solve(x0_val, max_iter=80):
        """Combines IQI, Newton, and bisection for robust convergence."""
        x = float(x0_val)
        fx = _safe_f(x)
        if not math.isfinite(fx):
            return None
        _update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x

        # Track last 3 points for IQI
        hist_x = [x]
        hist_fx = [fx]

        a, b = None, None
        fa, fb = None, None

        for i in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break

            x_new = None

            # Try IQI if we have 3 points
            if len(hist_x) >= 3:
                x3, fx3 = hist_x[-1], hist_fx[-1]
                x2, fx2 = hist_x[-2], hist_fx[-2]
                x1, fx1 = hist_x[-3], hist_fx[-3]
                x_iqi = _iqi_step(x1, fx1, x2, fx2, x3, fx3)
                if x_iqi is not None and math.isfinite(x_iqi):
                    # Accept IQI step only if it's in a reasonable range
                    scale = max(1.0, abs(x3))
                    if abs(x_iqi - x3) < 10.0 * scale:
                        x_new = x_iqi

            # If IQI didn't give a step, try Newton
            if x_new is None:
                x_new = _newton_step(x, fx)

            # If Newton didn't give a step, try bisection if bracket exists
            if x_new is None:
                if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                    if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                        x_new = (a + b) / 2.0
                    else:
                        break
                else:
                    break

            if x_new is None or not math.isfinite(x_new):
                break

            # Evaluate at new point
            fx_new = _safe_f(x_new)
            if not math.isfinite(fx_new):
                # Step failed; fall back to bisection if possible
                if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                    if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                        x_new = (a + b) / 2.0
                        fx_new = _safe_f(x_new)
                        if not math.isfinite(fx_new):
                            break
                    else:
                        break
                else:
                    break

            _update_best(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new

            # Update bracket
            if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
                a, b = x, x_new
                fa, fb = fx, fx_new
                if abs(fa) > abs(fb):
                    a, b = b, a
                    fa, fb = fb, fa

            # Update history for IQI (keep last 3)
            hist_x.append(x_new)
            hist_fx.append(fx_new)
            if len(hist_x) > 3:
                hist_x.pop(0)
                hist_fx.pop(0)

            x, fx = x_new, fx_new
            if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
                break

        return x if math.isfinite(x) else None

    # --- Phase 1: Try hybrid solver from x0 ---
    result = _hybrid_solve(x0)
    if result is not None and math.isfinite(result):
        fr = _safe_f(result)
        if math.isfinite(fr) and abs(fr) <= 1e-8:
            return result
        if math.isfinite(fr):
            _update_best(result, fr)

    # --- Phase 2: Scan for sign changes around x0 ---
    base = float(x0)
    scan_scale = max(1.0, abs(base))

    found_bracket = False
    for mult in [0.1, 0.2, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 25.0, 50.0, 100.0, 200.0, 500.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 10:
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _safe_f(a)
            fb = _safe_f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            _update_best(a, fa)
            _update_best(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisection(a, b, fa, fb, max_iter=80)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                found_bracket = True
                break
        if found_bracket:
            break

    # --- Phase 3: Local search for sign changes near best_x ---
    if not _time_ok() or _evals >= _MAX_EVALS - 10:
        return best_x

    center = best_x
    scale = max(1.0, abs(center))

    for mult in [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 10:
            break
        for sign in [-1, 1]:
            a = center - sign * mult * scale
            b = center + sign * mult * scale
            if a == b:
                continue
            fa = _safe_f(a)
            fb = _safe_f(b)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            _update_best(a, fa)
            _update_best(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisection(a, b, fa, fb, max_iter=60)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                break

    # --- Phase 4: Golden-section search on |f| as fallback ---
    if not _time_ok() or _evals >= _MAX_EVALS - 10:
        return best_x

    center = best_x
    scale = max(1.0, abs(center))
    a = center - 5.0 * scale
    b = center + 5.0 * scale

    for mult in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01]:
        a = center - mult * scale
        b = center + mult * scale
        ga = _safe_f(a)
        gb = _safe_f(b)
        if math.isfinite(ga) and math.isfinite(gb):
            break

    if math.isfinite(ga) and math.isfinite(gb):
        gr = (math.sqrt(5.0) - 1.0) / 2.0
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = _safe_f(c)
        fd = _safe_f(d)
        _update_best(c, fc)
        _update_best(d, fd)

        for _ in range(40):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            if abs(fc) <= 1e-8:
                return c
            if abs(fd) <= 1e-8:
                return d
            if abs(fc) < abs(fd):
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = _safe_f(c)
                _update_best(c, fc)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = _safe_f(d)
                _update_best(d, fd)
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        mid = (a + b) / 2.0
        fmid = _safe_f(mid)
        _update_best(mid, fmid)
        if abs(fmid) <= 1e-8:
            return mid

    # --- Phase 5: Final high-precision polish from best_x ---
    # Use tighter tolerance (1e-14) and secant/Newton hybrid for robustness
    if _time_ok() and _evals < _MAX_EVALS - 10:
        x = best_x
        fx = _safe_f(x)
        if not math.isfinite(fx):
            return best_x
        _update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x

        # Secant-based polish: track previous point for secant method
        x_prev = x + 1e-6 * max(1.0, abs(x))
        fx_prev = _safe_f(x_prev)
        if not math.isfinite(fx_prev):
            fx_prev = _safe_f(x - 1e-6 * max(1.0, abs(x)))
            if not math.isfinite(fx_prev):
                return best_x
            x_prev = x - 1e-6 * max(1.0, abs(x))

        for _ in range(40):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            if abs(fx) <= 1e-14:
                return x

            # Try Newton step
            d = _deriv(x, fx)
            x_new = None
            if d != 0 and math.isfinite(d):
                step = fx / d
                max_step = 2.0 * max(1.0, abs(x))
                if abs(step) > max_step:
                    step = max_step if step > 0 else -max_step
                x_new = x - step
                if not math.isfinite(x_new):
                    x_new = None

            # Fall back to secant
            if x_new is None and math.isfinite(fx_prev) and abs(x - x_prev) > 1e-15:
                denom = fx - fx_prev
                if abs(denom) > 1e-30:
                    x_new = x - fx * (x - x_prev) / denom
                    if not math.isfinite(x_new):
                        x_new = None

            if x_new is None or not math.isfinite(x_new):
                break

            fx_new = _safe_f(x_new)
            if not math.isfinite(fx_new):
                break

            _update_best(x_new, fx_new)
            if abs(fx_new) <= 1e-8:
                return x_new

            # Update secant history: keep the two most recent distinct points
            x_prev, fx_prev = x, fx
            x, fx = x_new, fx_new

            if abs(x_new - x) < 1e-15 * max(1.0, abs(x)):
                break

        # Final check
        if abs(fx) <= 1e-8:
            return x
        return best_x
# EVOLVE-BLOCK-END
