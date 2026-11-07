"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def _safe_f(f, x):
    """Call f(x) with safety; return None on exception or non-finite."""
    try:
        y = f(x)
        if y is None or not math.isfinite(y):
            return None
        return float(y)
    except Exception:
        return None

def find_root(f, x0):
    x = float(x0)
    deadline = time.monotonic() + 118.0
    best_x = x
    best_fx = _safe_f(f, x)

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return x

    # --- Phase 1: Secant/Newton hybrid (cheap, few calls) ---
    x_prev = x - 1e-4 * max(1.0, abs(x))
    fx_prev = _safe_f(f, x_prev)
    x_cur = x
    fx_cur = best_fx

    for _ in range(20):
        if time.monotonic() > deadline:
            break
        if fx_cur is not None and abs(fx_cur) <= 1e-10:
            return x_cur
        if fx_prev is not None and abs(fx_prev) <= 1e-10:
            return x_prev
        if fx_cur is None or fx_prev is None:
            break
        if fx_cur == fx_prev:
            break
        x_new = x_cur - fx_cur * (x_cur - x_prev) / (fx_cur - fx_prev)
        if not math.isfinite(x_new):
            break
        fx_new = _safe_f(f, x_new)
        if fx_new is None:
            break
        # Track best
        if abs(fx_new) < abs(best_fx) if best_fx is not None else True:
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new
        x_prev, fx_prev = x_cur, fx_cur
        x_cur, fx_cur = x_new, fx_new

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 2: Bisection with adaptive bracketing ---
    scale = max(1.0, abs(x0))
    left = x0 - scale
    right = x0 + scale
    f_left = _safe_f(f, left)
    f_right = _safe_f(f, right)

    if f_left is None or f_right is None or f_left * f_right > 0:
        for expand_factor in [2, 4, 8, 16, 32, 64, 128]:
            if time.monotonic() > deadline:
                break
            left = x0 - scale * expand_factor
            right = x0 + scale * expand_factor
            f_left = _safe_f(f, left)
            f_right = _safe_f(f, right)
            if f_left is None or f_right is None:
                continue
            if f_left * f_right <= 0:
                break

    if f_left is not None and f_right is not None and f_left * f_right <= 0:
        for _ in range(50):
            if time.monotonic() > deadline:
                break
            mid = (left + right) / 2.0
            f_mid = _safe_f(f, mid)
            if f_mid is None:
                break
            if abs(f_mid) <= 1e-10:
                return mid
            if f_mid * f_left <= 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid
            if abs(f_mid) < abs(best_fx) if best_fx is not None else True:
                best_x = mid
                best_fx = f_mid
            if abs(f_mid) <= 1e-8:
                return mid

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # --- Phase 3: High-precision polish (central difference derivative) ---
    x_p = best_x
    fx_p = _safe_f(f, x_p)
    if fx_p is None or abs(fx_p) > 1e-8:
        return x_p  # Can't polish further

    for _ in range(15):
        if time.monotonic() > deadline:
            break
        if abs(fx_p) <= 1e-12:
            return x_p
        h = 1e-8 * max(1.0, abs(x_p))
        f_plus = _safe_f(f, x_p + h)
        f_minus = _safe_f(f, x_p - h)
        if f_plus is None or f_minus is None:
            break
        d = (f_plus - f_minus) / (2.0 * h)
        if d == 0 or not math.isfinite(d):
            break
        x_new = x_p - fx_p / d
        if not math.isfinite(x_new):
            break
        fx_new = _safe_f(f, x_new)
        if fx_new is None:
            break
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        if abs(fx_new) <= 1e-8:
            return x_new
        x_p, fx_p = x_new, fx_new

    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x

    # Final safety: ensure finite return
    if not math.isfinite(best_x):
        return float(x0)
    return best_x
# EVOLVE-BLOCK-END
