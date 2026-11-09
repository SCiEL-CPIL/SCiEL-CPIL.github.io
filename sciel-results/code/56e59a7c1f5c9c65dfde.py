"""Direct root finder for smooth polynomials using a tight secant + safeguarded Newton."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 145.0  # leave a margin before 150s kill
_SAFETY = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    tol = 1e-8
    max_calls = 480
    calls = 0

    def ceval(x):
        """Counted, safe evaluation."""
        nonlocal calls
        if calls >= max_calls:
            return None, False
        calls += 1
        try:
            val = f(x)
        except Exception:
            return None, False
        if not _is_finite(val):
            return None, False
        return val, True

    # --- Phase 0: initial evaluation at x0 ---
    x = float(x0)
    fx, ok = ceval(x)
    if not ok:
        # probe a small grid around x0
        for dx in (1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0):
            xt = x + dx
            fx, ok = ceval(xt)
            if ok:
                x = xt
                break
        if not ok:
            return x

    if abs(fx) <= tol:
        return x

    # --- Phase 1: Secant with a second point, then iterate ---
    h0 = 1e-4 * max(1.0, abs(x))
    x_prev = x
    fx_prev = fx
    x_curr = x + h0
    fx_curr, ok = ceval(x_curr)
    if not ok:
        x_curr = x - h0
        fx_curr, ok = ceval(x_curr)
    if not ok:
        # last resort: just return best
        return x

    if abs(fx_curr) <= tol:
        return x_curr

    best_x = x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
    best_f = min(abs(fx_curr), abs(fx_prev))

    # Secant iterations: cap at ~40 to keep within budget
    for _ in range(40):
        if calls >= max_calls or _time_left() < 0:
            break
        if fx_curr is None or fx_prev is None:
            break
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-18:
            # Try a small nudge
            x_curr += 1e-8 * max(1.0, abs(x_curr))
            fx_curr, ok = ceval(x_curr)
            if not ok:
                break
            continue
        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        if not _is_finite(x_next):
            break
        fx_next, ok = ceval(x_next)
        if not ok:
            break
        if abs(fx_next) < best_f:
            best_f = abs(fx_next)
            best_x = x_next
        if abs(fx_next) <= tol:
            return x_next
        # Divergence guard: if x jumps very far and |f| grows, bail
        if abs(x_next) > 1e12 * max(1.0, abs(x_curr)):
            break
        x_prev, fx_prev = x_curr, fx_curr
        x_curr, fx_curr = x_next, fx_next

    if best_f <= tol:
        return best_x

    # --- Phase 2: Secant-Newton hybrid with finite-difference derivative, safeguarded ---
    # Start from best point. We'll do up to ~15 iterations, each costing up to 3 calls.
    xw = best_x
    for _ in range(15):
        if calls >= max_calls or _time_left() < 0:
            break
        f_w, ok = ceval(xw)
        if not ok:
            break
        if abs(f_w) <= tol:
            return x_w
        if abs(f_w) < best_f:
            best_f = abs(f_w)
            best_x = x_w

        # Compute derivative via central diff: 2 calls
        h = 1e-6 * max(1.0, abs(xw))
        fp, ok1 = ceval(xw + h)
        fm, ok2 = ceval(xw - h)
        if not (ok1 and ok2):
            # Fall back to one-sided if one fails
            if ok1:
                d = (fp - f_w) / h
            elif ok2:
                d = (f_w - fm) / h
            else:
                break
        else:
            d = (fp - fm) / (2 * h)

        if abs(d) < 1e-18:
            break

        x_newton = xw - f_w / d
        if not _is_finite(x_newton):
            break

        # Safeguard: if newton step is too far, halve it
        step = x_newton - xw
        max_step = 100.0 * max(1.0, abs(xw))
        if abs(step) > max_step:
            x_newton = xw + max_step * math.copysign(1.0, step)

        f_newton, ok = ceval(x_newton)
        if not ok:
            break
        if abs(f_newton) <= tol:
            return x_newton
        if abs(f_newton) < best_f:
            best_f = abs(f_newton)
            best_x = x_newton
        xw = x_newton

    if best_f <= tol:
        return best_x

    # --- Phase 3: Bisection fallback using sign change search ---
    # Try to find a bracket around best_x
    x_lo, x_hi = best_x, best_x
    f_lo = None
    f_hi = None
    # We don't have f(best_x) cached reliably; re-evaluate
    f_lo, ok = ceval(x_lo)
    if ok:
        f_hi = f_lo
    step = 0.1 * max(1.0, abs(best_x))
    bracketed = False
    for _ in range(30):
        if calls >= max_calls or _time_left() < 0:
            break
        # Expand left
        x_try = x_lo - step
        f_try, ok = ceval(x_try)
        if ok:
            if f_lo is not None and f_lo * f_try < 0:
                x_hi = x_lo
                x_lo = x_try
                f_lo = f_try
                f_hi = f_lo if f_lo is not None else 0
                # re-evaluate hi
                f_hi, _ = ceval(x_hi)
                bracketed = True
                break
            if f_hi is not None and f_hi * f_try < 0:
                x_hi = x_hi if x_hi != x_lo else best_x
                x_lo = x_try
                f_lo = f_try
                bracketed = True
                break
        # Expand right
        x_try = x_hi + step
        f_try, ok = ceval(x_try)
        if ok:
            if f_lo is not None and f_lo * f_try < 0:
                x_hi = x_try
                f_hi = f_try
                bracketed = True
                break
            if f_hi is not None and f_hi * f_try < 0:
                x_lo = x_hi
                f_lo = f_hi
                x_hi = x_try
                f_hi = f_try
                bracketed = True
                break
        step *= 2.0

    if bracketed:
        for _ in range(60):
            if calls >= max_calls or _time_left() < 0:
                break
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid, ok = ceval(x_mid)
            if not ok:
                break
            if abs(f_mid) <= tol:
                return x_mid
            if abs(f_mid) < best_f:
                best_f = abs(f_mid)
                best_x = x_mid
            if f_lo * f_mid <= 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid

    # --- Phase 4: Final high-precision polish pass ---
    # If we're close (|f| < 1e-4), do a few more secant/Newton steps with tighter tolerance
    if best_f < 1e-4 and calls < max_calls - 5:
        xp = best_x
        fp, ok = ceval(xp)
        if ok and abs(fp) < best_f:
            best_f = abs(fp)
            best_x = xp
        if ok:
            # Try a few more secant steps with tighter tol
            h = 1e-5 * max(1.0, abs(xp))
            x2 = xp + h
            f2, ok2 = ceval(x2)
            if ok2:
                for _ in range(10):
                    if calls >= max_calls or _time_left() < 0:
                        break
                    denom = f2 - fp
                    if abs(denom) < 1e-20:
                        break
                    xn = x2 - f2 * (x2 - xp) / denom
                    if not _is_finite(xn):
                        break
                    fn, okn = ceval(xn)
                    if not okn:
                        break
                    if abs(fn) < best_f:
                        best_f = abs(fn)
                        best_x = xn
                    if abs(fn) <= tol:
                        return xn
                    xp, fp = x2, f2
                    x2, f2 = xn, fn

    return best_x
