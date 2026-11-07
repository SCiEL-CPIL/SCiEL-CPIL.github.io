import math
import time

try:
    from scipy.optimize import brentq as _scipy_brentq
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

_DEADLINE = time.time() + 110.0  # Safety margin for 120s limit
_CALL_BUDGET = 480  # Safety margin for 500 limit

def _time_left():
    return _DEADLINE - time.time()

def _safe_call(f, x):
    try:
        val = f(x)
        if not math.isfinite(val):
            return float('nan')
        return val
    except Exception:
        return float('nan')

def find_root(f, x0):
    counter = [0]
    x0 = float(x0)
    
    # Initial call
    fx0 = _safe_call(f, x0)
    counter[0] += 1
    if math.isnan(fx0):
        return x0
    if abs(fx0) <= 1e-8:
        return x0

    best_x = x0
    best_fx = fx0
    best_abs = abs(fx0)

    # Strategy 1: Secant Method
    # Start with a small perturbation
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = _safe_call(f, x1)
    counter[0] += 1
    
    if math.isnan(fx1):
        # If perturbation failed, try a larger one
        x1 = x0 + 1e-2 * max(1.0, abs(x0))
        fx1 = _safe_call(f, x1)
        counter[0] += 1
        if math.isnan(fx1):
            return best_x

    if abs(fx1) < best_abs:
        best_x = x1
        best_fx = fx1
        best_abs = abs(fx1)
    if abs(fx1) <= 1e-8:
        return x1

    bracket = None
    x_prev = x0
    fx_prev = fx0
    x_curr = x1
    fx_curr = fx1

    # Run Secant for up to 25 iterations
    for _ in range(25):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
            
        # Check for sign change (bracket)
        if fx_prev * fx_curr < 0:
            bracket = (x_prev, x_curr, fx_prev, fx_curr)
            break
            
        # Calculate next point
        denom = fx_curr - fx_prev
        if denom == 0:
            # Degenerate case, perturb
            x_next = x_curr + 1e-4 * max(1.0, abs(x_curr))
        else:
            x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
            
        if not math.isfinite(x_next):
            break
            
        fx_next = _safe_call(f, x_next)
        counter[0] += 1
        
        if math.isnan(fx_next):
            break
            
        if abs(fx_next) < best_abs:
            best_x = x_next
            best_fx = fx_next
            best_abs = abs(fx_next)
            
        if abs(fx_next) <= 1e-8:
            return x_next
            
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next

    # If we have a bracket, use Brent's method (or Bisection)
    if bracket is not None:
        a, b, fa, fb = bracket
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        if _HAS_SCIPY:
            try:
                # scipy brentq is very efficient
                x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                # Verify
                fx_br = _safe_call(f, x_br)
                counter[0] += 1
                if not math.isnan(fx_br) and abs(fx_br) <= 1e-8:
                    return x_br
                if not math.isnan(fx_br) and abs(fx_br) < best_abs:
                    best_x = x_br
                    best_fx = fx_br
                    best_abs = abs(fx_br)
            except Exception:
                pass
        
        # Fallback: Bisection
        if best_abs > 1e-8:
            for _ in range(50):
                if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                    break
                if fa * fb > 0:
                    break
                c = (a + b) / 2.0
                fc = _safe_call(f, c)
                counter[0] += 1
                if math.isnan(fc):
                    break
                if abs(fc) < best_abs:
                    best_x = c
                    best_fx = fc
                    best_abs = abs(fc)
                if abs(fc) <= 1e-8:
                    return c
                if fa * fc <= 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
                if abs(b - a) < 1e-12 * max(1.0, abs(c)):
                    break

    # If no bracket was found during Secant, try to find one around best_x
    if best_abs > 1e-8:
        # Expand search for a bracket
        step = 1e-3 * max(1.0, abs(best_x))
        center = best_x
        fc = best_fx
        
        for i in range(20):
            if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                break
            
            # Check right
            xr = center + step
            fr = _safe_call(f, xr)
            counter[0] += 1
            if not math.isnan(fr):
                if abs(fr) < best_abs:
                    best_x = xr
                    best_fx = fr
                    best_abs = abs(fr)
                if abs(fr) <= 1e-8:
                    return xr
                if fc * fr < 0:
                    # Found bracket [center, xr]
                    a, b = center, xr
                    fa, fb = fc, fr
                    if _HAS_SCIPY:
                        try:
                            x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                            fx_br = _safe_call(f, x_br)
                            counter[0] += 1
                            if not math.isnan(fx_br) and abs(fx_br) <= 1e-8:
                                return x_br
                        except Exception:
                            pass
                    # Bisection fallback
                    for _ in range(30):
                        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                            break
                        if fa * fb > 0:
                            break
                        c = (a + b) / 2.0
                        fc_mid = _safe_call(f, c)
                        counter[0] += 1
                        if math.isnan(fc_mid):
                            break
                        if abs(fc_mid) < best_abs:
                            best_x = c
                            best_fx = fc_mid
                            best_abs = abs(fc_mid)
                        if abs(fc_mid) <= 1e-8:
                            return c
                        if fa * fc_mid <= 0:
                            b = c
                            fb = fc_mid
                        else:
                            a = c
                            fa = fc_mid
                    break

            # Check left
            xl = center - step
            fl = _safe_call(f, xl)
            counter[0] += 1
            if not math.isnan(fl):
                if abs(fl) < best_abs:
                    best_x = xl
                    best_fx = fl
                    best_abs = abs(fl)
                if abs(fl) <= 1e-8:
                    return xl
                if fc * fl < 0:
                    # Found bracket [xl, center]
                    a, b = xl, center
                    fa, fb = fl, fc
                    if _HAS_SCIPY:
                        try:
                            x_br = _scipy_brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                            fx_br = _safe_call(f, x_br)
                            counter[0] += 1
                            if not math.isnan(fx_br) and abs(fx_br) <= 1e-8:
                                return x_br
                        except Exception:
                            pass
                    # Bisection fallback
                    for _ in range(30):
                        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
                            break
                        if fa * fb > 0:
                            break
                        c = (a + b) / 2.0
                        fc_mid = _safe_call(f, c)
                        counter[0] += 1
                        if math.isnan(fc_mid):
                            break
                        if abs(fc_mid) < best_abs:
                            best_x = c
                            best_fx = fc_mid
                            best_abs = abs(fc_mid)
                        if abs(fc_mid) <= 1e-8:
                            return c
                        if fa * fc_mid <= 0:
                            b = c
                            fb = fc_mid
                        else:
                            a = c
                            fa = fc_mid
                    break

            step *= 2.0

    return best_x
