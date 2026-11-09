"""Model-and-optimize root finder for ill-conditioned equations."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    _t0 = time.time()
    _TIME_LIMIT = 115.0
    _MAX_EVALS = 480
    _evals = [0]

    def _f(x):
        _evals[0] += 1
        try:
            return float(f(x))
        except Exception:
            return float('nan')

    def _time_ok():
        return (time.time() - _t0) < _TIME_LIMIT

    def _budget_ok():
        return _evals[0] < _MAX_EVALS

    def _safe(x):
        return math.isfinite(x)

    best_x = float(x0)
    best_fx = _f(best_x)
    best_abs = abs(best_fx) if _safe(best_fx) else float('inf')

    def _update(x, fx):
        nonlocal best_x, best_fx, best_abs
        if not _safe(fx):
            return False
        af = abs(fx)
        if af < best_abs:
            best_abs = af
            best_x = float(x)
            best_fx = float(fx)
            return af <= 1e-8
        return False

    _update(best_x, best_fx)
    if best_abs <= 1e-8:
        return best_x

    def _iqi(x1, f1, x2, f2, x3, f3):
        d12 = x1 - x2
        d13 = x1 - x3
        d23 = x2 - x3
        if abs(d12) < 1e-15 or abs(d13) < 1e-15 or abs(d23) < 1e-15:
            return None
        c2 = f1/(d12*d13) + f2/((-d12)*d23) + f3/((-d13)*(-d23))
        c1 = -f1*(x2+x3)/(d12*d13) - f2*(x1+x3)/((-d12)*d23) - f3*(x1+x2)/((-d13)*(-d23))
        c0 = f1*x2*x3/(d12*d13) + f2*x1*x3/((-d12)*d23) + f3*x1*x2/((-d13)*(-d23))
        if abs(c2) < 1e-30:
            if abs(c1) < 1e-30:
                return None
            x_root = -c0/c1
        else:
            disc = c1*c1 - 4*c2*c0
            if disc < 0:
                return None
            sqrt_disc = math.sqrt(disc)
            if abs(c1) < 1e-30:
                x_root1 = sqrt_disc / (2*abs(c2))
                x_root2 = -sqrt_disc / (2*abs(c2))
            else:
                q = -0.5 * (c1 + math.copysign(sqrt_disc, c1))
                x_root1 = q / c2
                x_root2 = c0 / q
            if abs(x_root1 - x3) < abs(x_root2 - x3):
                x_root = x_root1
            else:
                x_root = x_root2
        if not _safe(x_root):
            return None
        return x_root

    def _secant(x1, f1, x2, f2):
        if abs(f2 - f1) < 1e-30:
            return None
        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
        if not _safe(x_new):
            return None
        return x_new

    def _bisect(a, b, fa, fb, max_iter=60):
        for _ in range(max_iter):
            if not _time_ok() or not _budget_ok():
                break
            mid = (a + b) / 2.0
            fmid = _f(mid)
            if not _safe(fmid):
                break
            _update(mid, fmid)
            if abs(fmid) <= 1e-8:
                return mid
            if (fmid <= 0 and fa <= 0) or (fmid >= 0 and fa >= 0):
                a, fa = mid, fmid
            else:
                b, fb = mid, fmid
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        return (a + b) / 2.0

    x = float(x0)
    fx = _f(x)
    if not _safe(fx):
        fx = best_fx
        x = best_x
    _update(x, fx)

    if abs(fx) <= 1e-8:
        return x

    scale = max(1.0, abs(x))
    h = 1e-4 * scale
    x2 = x + h
    fx2 = _f(x2)
    if not _safe(fx2):
        h = -h
        x2 = x + h
        fx2 = _f(x2)
    if not _safe(fx2):
        # Fallback: scan
        base = best_x
        scan_scale = max(1.0, abs(base))
        for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
            if not _time_ok() or not _budget_ok():
                break
            for sign in [-1, 1]:
                a = base - sign * mult * scan_scale
                b = base + sign * mult * scan_scale
                if a == b:
                    continue
                fa = _f(a)
                fb = _f(b)
                if not (_safe(fa) and _safe(fb)):
                    continue
                _update(a, fa)
                _update(b, fb)
                if abs(fa) <= 1e-8:
                    return a
                if abs(fb) <= 1e-8:
                    return b
                if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                    res = _bisect(a, b, fa, fb, max_iter=50)
                    fres = _f(res)
                    _update(res, fres)
                    if abs(fres) <= 1e-8:
                        return res
                    return res
        return best_x

    _update(x2, fx2)
    if abs(fx2) <= 1e-8:
        return x2

    hist_x = [x, x2]
    hist_f = [fx, fx2]

    bracket_a = bracket_b = None
    bracket_fa = bracket_fb = None

    for i in range(80):
        if not _time_ok() or not _budget_ok():
            break

        x_curr = hist_x[-1]
        fx_curr = hist_f[-1]

        if abs(fx_curr) <= 1e-8:
            return x_curr

        x_new = None

        if len(hist_x) >= 3:
            x_iqi = _iqi(hist_x[-3], hist_f[-3], hist_x[-2], hist_f[-2], hist_x[-1], hist_f[-1])
            if x_iqi is not None and _safe(x_iqi):
                if abs(x_iqi - x_curr) < 10.0 * max(1.0, abs(x_curr)):
                    x_new = x_iqi

        if x_new is None and len(hist_x) >= 2:
            x_sec = _secant(hist_x[-2], hist_f[-2], hist_x[-1], hist_f[-1])
            if x_sec is not None and _safe(x_sec):
                if abs(x_sec - x_curr) < 10.0 * max(1.0, abs(x_curr)):
                    x_new = x_sec

        if x_new is None:
            if len(hist_x) >= 2:
                if abs(hist_f[-1]) < abs(hist_f[-2]):
                    x_new = x_curr + 0.5 * (x_curr - hist_x[-2])
                else:
                    x_new = x_curr - 0.5 * (x_curr - hist_x[-2])
            else:
                x_new = x_curr + 0.5 * scale

        fx_new = _f(x_new)
        if not _safe(fx_new):
            x_new = x_curr + 0.1 * (x_curr - hist_x[-2] if len(hist_x) >= 2 else scale)
            fx_new = _f(x_new)
            if not _safe(fx_new):
                break

        _update(x_new, fx_new)
        if abs(fx_new) <= 1e-8:
            return x_new

        hist_x.append(x_new)
        hist_f.append(fx_new)
        if len(hist_x) > 3:
            hist_x.pop(0)
            hist_f.pop(0)

        if len(hist_x) >= 2:
            x_a, x_b = hist_x[-2], hist_x[-1]
            f_a, f_b = hist_f[-2], hist_f[-1]
            if (f_a <= 0 and f_b >= 0) or (f_a >= 0 and f_b <= 0):
                bracket_a, bracket_b = x_a, x_b
                bracket_fa, bracket_fb = f_a, f_b

        if abs(x_new - x_curr) < 1e-14 * max(1.0, abs(x_new)):
            break

    # Phase 2: If we have a bracket, use bisection/IQI hybrid
    if bracket_a is not None and bracket_b is not None:
        if (bracket_fa <= 0 and bracket_fb >= 0) or (bracket_fa >= 0 and bracket_fb <= 0):
            a, b = bracket_a, bracket_b
            fa, fb = bracket_fa, bracket_fb
            x_mid = (a + b) / 2.0
            fmid = _f(x_mid)
            if not _safe(fmid):
                fmid = fa if abs(fa) < abs(fb) else fb
                x_mid = a if abs(fa) < abs(fb) else b
            _update(x_mid, fmid)
            if abs(fmid) <= 1e-8:
                return x_mid

            for i in range(50):
                if not _time_ok() or not _budget_ok():
                    break

                x_iqi = _iqi(a, fa, x_mid, fmid, b, fb)
                x_try = None
                if x_iqi is not None and _safe(x_iqi) and a < x_iqi < b:
                    x_try = x_iqi
                else:
                    x_try = (a + b) / 2.0

                fx_try = _f(x_try)
                if not _safe(fx_try):
                    x_try = (a + b) / 2.0
                    fx_try = _f(x_try)
                    if not _safe(fx_try):
                        break

                _update(x_try, fx_try)
                if abs(fx_try) <= 1e-8:
                    return x_try

                if (fx_try <= 0 and fa <= 0) or (fx_try >= 0 and fa >= 0):
                    a, fa = x_try, fx_try
                else:
                    b, fb = x_try, fx_try

                x_mid = (a + b) / 2.0
                fmid = _f(x_mid)
                if not _safe(fmid):
                    break
                _update(x_mid, fmid)
                if abs(fmid) <= 1e-8:
                    return x_mid

                if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                    break

            res = (a + b) / 2.0
            fres = _f(res)
            _update(res, fres)
            if abs(fres) <= 1e-8:
                return res

    # Phase 3: Scan for brackets
    if not _time_ok() or not _budget_ok():
        return best_x

    base = best_x
    scan_scale = max(1.0, abs(base))
    found = False
    for mult in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]:
        if not _time_ok() or not _budget_ok():
            break
        for sign in [-1, 1]:
            a = base - sign * mult * scan_scale
            b = base + sign * mult * scan_scale
            if a == b:
                continue
            fa = _f(a)
            fb = _f(b)
            if not (_safe(fa) and _safe(fb)):
                continue
            _update(a, fa)
            _update(b, fb)
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if (fa <= 0 and fb >= 0) or (fa >= 0 and fb <= 0):
                res = _bisect(a, b, fa, fb, max_iter=50)
                fres = _f(res)
                _update(res, fres)
                if abs(fres) <= 1e-8:
                    return res
                found = True
                break
        if found:
            break

    # Phase 4: Final secant polish
    if _time_ok() and _budget_ok() and len(hist_x) >= 2:
        x1, f1 = hist_x[-2], hist_f[-2]
        x2, f2 = hist_x[-1], hist_f[-1]
        for _ in range(15):
            if not _time_ok() or not _budget_ok():
                break
            if abs(f2) <= 1e-8:
                return x2
            x_new = _secant(x1, f1, x2, f2)
            if x_new is None:
                break
            f_new = _f(x_new)
            if not _safe(f_new):
                break
            _update(x_new, f_new)
            if abs(f_new) <= 1e-8:
                return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

    return best_x

# EVOLVE-BLOCK-END
