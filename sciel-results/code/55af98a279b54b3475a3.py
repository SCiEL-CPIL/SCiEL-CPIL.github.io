"""Robust ill-conditioned root finder combining efficient Newton + adaptive bracketing."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 115.0
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
            return float(_f(x))
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

    # --- Efficient derivative: single h, central then forward ---
    def _deriv(x, fx):
        scale = max(1.0, abs(x))
        h = 1e-6 * scale
        if h == 0:
            h = 1e-6
        fm = _safe_f(x - h)
        fp = _safe_f(x + h)
        if math.isfinite(fm) and math.isfinite(fp):
            d = (fp - fm) / (2.0 * h)
            if math.isfinite(d) and abs(d) > 1e-30:
                return d
        # Forward difference
        h2 = 1e-5 * scale
        if h2 == 0:
            h2 = 1e-5
        fp2 = _safe_f(x + h2)
        if math.isfinite(fp2) and math.isfinite(fx):
            d = (fp2 - fx) / h2
            if math.isfinite(d) and abs(d) > 1e-30:
                return d
        return 0.0

    # --- Newton step with safeguard ---
    def _newton_step(x, fx):
        d = _deriv(x, fx)
        if d == 0 or not math.isfinite(d):
            return None
        step = fx / d
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            step = max_step if step > 0 else -max_step
        x_new = x - step
        if not math.isfinite(x_new):
            return None
        return x_new

    # --- Bisection on bracket [a, b] ---
    def _bisection(a, b, fa, fb, max_iter=100):
        if not _time_ok():
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
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return (a + b) / 2.0

    # --- Robust Brent's method (clean implementation) ---
    def _brent(a, b, fa, fb, max_iter=150):
        if not _time_ok():
            return None
        # Ensure |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c, fc = a, fa
        d = b - a
        mflag = True
        for _ in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            # Invariant: f(a)*f(b) <= 0
            if (fc <= 0 and fa <= 0) or (fc >= 0 and fa >= 0):
                c, fc = b, fb
                d = b - a
                mflag = True
            if abs(fc) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
                c, fc = b, a
                fc = fa
                d = b - a
                mflag = True
            if abs(fc) <= 1e-8:
                return c
            if abs(d) < 1e-15 * max(1.0, abs(a)):
                return a
            if mflag:
                mflag = False
                if fa != fb:
                    s = a - fa * (b - a) / (fb - fa)
                else:
                    s = (a + b) / 2.0
                if not math.isfinite(s):
                    s = (a + b) / 2.0
            else:
                # Inverse quadratic interpolation
                if fa == fb:
                    s = (a + b) / 2.0
                elif fa == fc:
                    s = a - fa * (b - a) / (fb - fa)
                else:
                    A = (b - a) * (fb - fc)
                    B = (a - b) * (fa - fc)
                    C = (a - c) * (fa - fb)
                    denom = A * B - C * (B + A)
                    if denom == 0:
                        s = (a + b) / 2.0
                    else:
                        s = a + C * (A - B) / denom
                        if not math.isfinite(s):
                            s = (a + b) / 2.0
                if (s <= a or s >= b) or not math.isfinite(s):
                    s = (a + b) / 2.0
            d = b - a
            fs = _safe_f(s)
            if not math.isfinite(fs):
                break
            _update_best(s, fs)
            if abs(fs) <= 1e-8:
                return s
            if (fs <= 0 and fa <= 0) or (fs >= 0 and fa >= 0):
                a = s
                fa = fs
            else:
                b = s
                fb = fs
        return (a + b) / 2.0

    # --- Newton with adaptive bracketing and bisection fallback ---
    def _newton_with_bisection(x0_val, max_iter=100):
        x = float(x0_val)
        fx = _safe_f(x)
        if not math.isfinite(fx):
            return None
        _update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x
        a, b = None, None
        fa, fb = None, None
        prev_x = None
        for i in range(max_iter):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            x_new = _newton_step(x, fx)
            if x_new is not None and math.isfinite(x_new):
                fx_new = _safe_f(x_new)
                if not math.isfinite(fx_new):
                    break
                _update_best(x_new, fx_new)
                if abs(fx_new) <= 1e-8:
                    return x_new
                # Set/update bracket
                if (fx_new <= 0 and fx >= 0) or (fx_new >= 0 and fx <= 0):
                    a, b = x, x_new
                    fa, fb = fx, fx_new
                elif a is not None and b is not None:
                    if (fx_new <= 0 and fa <= 0) or (fx_new >= 0 and fa >= 0):
                        a, fa = x_new, fx_new
                    else:
                        b, fb = x_new, fx_new
                x, fx = x_new, fx_new
                if prev_x is not None and abs(x_new - prev_x) < 1e-14 * max(1.0, abs(x)):
                    break
                prev_x = x
            else:
                # Newton failed, use bisection if bracket exists
                if a is not None and b is not None and math.isfinite(fa) and math.isfinite(fb):
                    if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
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
                        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                            break
                        x, fx = mid, fmid
                        prev_x = None
                    else:
                        break
                else:
                    break
        return x if math.isfinite(x) else None

    # --- Phase 1: Newton from x0 ---
    result = _newton_with_bisection(x0)
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
    for mult in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]:
        if not _time_ok():
            break
        if _evals >= _MAX_EVALS - 20:
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
                res = _brent(a, b, fa, fb, max_iter=150)
                if res is not None and math.isfinite(res):
                    fres = _safe_f(res)
                    if math.isfinite(fres) and abs(fres) <= 1e-8:
                        return res
                    _update_best(res, fres)
                found_bracket = True
                break
        if found_bracket:
            break

    # --- Phase 3: Golden-section minimization of |f| ---
    if not _time_ok() or _evals >= _MAX_EVALS - 20:
        return best_x

    center = best_x
    scale = max(1.0, abs(center))
    a = center - 10.0 * scale
    b = center + 10.0 * scale
    ga = _safe_f(a)
    gb = _safe_f(b)
    if not (math.isfinite(ga) and math.isfinite(gb)):
        for mult in [5.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05]:
            if not _time_ok() or _evals >= _MAX_EVALS - 15:
                break
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
        for _ in range(80):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 10:
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
            if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
                break
        # Newton polish from golden-section best
        if _time_ok() and _evals < _MAX_EVALS - 10:
            x = best_x
            for _ in range(15):
                if not _time_ok():
                    break
                if _evals >= _MAX_EVALS - 5:
                    break
                fx = _safe_f(x)
                if not math.isfinite(fx):
                    break
                _update_best(x, fx)
                if abs(fx) <= 1e-8:
                    return x
                x_new = _newton_step(x, fx)
                if x_new is None or not math.isfinite(x_new):
                    break
                x = x_new

    # --- Phase 4: Final Newton polish ---
    if _time_ok() and _evals < _MAX_EVALS - 10:
        x = best_x
        for _ in range(20):
            if not _time_ok():
                break
            if _evals >= _MAX_EVALS - 5:
                break
            fx = _safe_f(x)
            if not math.isfinite(fx):
                break
            _update_best(x, fx)
            if abs(fx) <= 1e-8:
                return x
            x_new = _newton_step(x, fx)
            if x_new is None or not math.isfinite(x_new):
                break
            x = x_new

    return best_x
# EVOLVE-BLOCK-END
