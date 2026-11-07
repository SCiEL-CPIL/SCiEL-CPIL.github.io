"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Optimized: minimal bracket search + fast convergence + final polish.
    """
    start_time = time.time()
    deadline = start_time + 118.0
    TOL = 1e-9
    MAX_CALLS = 495
    POLISH_BUDGET = 20  # reserve calls for final high-precision polish

    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def _check_time():
        return time.time() < deadline

    def _safe_f(x_val):
        nonlocal call_count, best_x, best_f
        if call_count >= MAX_CALLS:
            return None
        try:
            val = f(x_val)
            call_count += 1
            if not math.isfinite(val):
                return None
            if abs(val) < abs(best_f):
                best_f = val
                best_x = x_val
            return val
        except Exception:
            call_count += 1
            return None

    x = float(x0)
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Phase 1: Efficient bracket search with minimal calls
    a, b, fa, fb = None, None, None, None

    # Try small steps first, then expand. Stop early when bracket found.
    # Use a list of candidate steps to try in order
    steps = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0]

    for s in steps:
        if not _check_time() or call_count >= MAX_CALLS - POLISH_BUDGET - 10:
            break

        # Try right
        xr = x + s
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

        # Try left
        xl = x - s
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

    # Phase 2: If bracket found, use efficient hybrid method
    if a is not None and b is not None and fa is not None and fb is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Store all evaluated points for secant use
        pts = [(x_lo, f_lo), (x_hi, f_hi)]

        # Main loop: alternate between secant (fast) and bisection (safe)
        for _ in range(60):
            if not _check_time() or call_count >= MAX_CALLS - POLISH_BUDGET:
                break

            if abs(f_lo) <= TOL:
                break
            if abs(f_hi) <= TOL:
                break

            # If interval is very small, just bisect
            if abs(x_hi - x_lo) < 1e-12 * max(1.0, abs(x_lo), abs(x_hi)):
                x_mid = (x_lo + x_hi) / 2.0
                f_mid = _safe_f(x_mid)
                if f_mid is not None:
                    if abs(f_mid) <= TOL:
                        break
                    if f_mid * f_lo < 0:
                        x_hi, f_hi = x_mid, f_mid
                    else:
                        x_lo, f_lo = x_mid, f_mid
                continue

            # Try secant step using last two points
            if len(pts) >= 2:
                x1, f1 = pts[-1]
                x2, f2 = pts[-2]
                if abs(f1 - f2) > 1e-15:
                    x_sec = x1 - f1 * (x1 - x2) / (f1 - f2)
                    # Check if secant point is within bracket
                    if x_lo < x_sec < x_hi:
                        f_sec = _safe_f(x_sec)
                        if f_sec is not None:
                            if abs(f_sec) <= TOL:
                                break
                            pts.append((x_sec, f_sec))
                            if f_sec * f_lo < 0:
                                x_hi, f_hi = x_sec, f_sec
                            else:
                                x_lo, f_lo = x_sec, f_sec
                            continue

            # Bisection
            x_mid = (x_lo + x_hi) / 2.0
            f_mid = _safe_f(x_mid)
            if f_mid is None:
                break
            if abs(f_mid) <= TOL:
                break
            pts.append((x_mid, f_mid))
            if f_mid * f_lo < 0:
                x_hi, f_hi = x_mid, f_mid
            else:
                x_lo, f_lo = x_mid, f_mid

        # Pick the better endpoint
        if abs(f_lo) < abs(f_hi):
            x_approx = x_lo
            f_approx = f_lo
        else:
            x_approx = x_hi
            f_approx = f_hi

        # Phase 3: Final polish using remaining budget
        x_curr = x_approx
        f_curr = f_approx

        # Use secant with the two best points for polish
        if len(pts) >= 2:
            # Find the two points closest to root
            pts_sorted = sorted(pts, key=lambda p: abs(p[1]))
            p1, p2 = pts_sorted[0], pts_sorted[1]
            for _ in range(POLISH_BUDGET):
                if not _check_time() or call_count >= MAX_CALLS:
                    break
                if abs(f_curr) <= TOL:
                    break
                if abs(p1[1] - p2[1]) > 1e-15:
                    x_new = p1[0] - p1[1] * (p1[0] - p2[0]) / (p1[1] - p2[1])
                    if math.isfinite(x_new):
                        f_new = _safe_f(x_new)
                        if f_new is not None:
                            if abs(f_new) <= TOL:
                                break
                            p2 = p1
                            p1 = (x_new, f_new)
                            x_curr, f_curr = x_new, f_new
                else:
                    break

        return x_curr

    # Phase 2b: No bracket found - use Newton with minimal derivative evals
    x_curr = float(x0)
    f_curr = fx

    for _ in range(40):
        if not _check_time() or call_count >= MAX_CALLS - POLISH_BUDGET:
            break

        if abs(f_curr) <= TOL:
            break

        # One-sided finite difference (1 eval)
        h = 1e-7 * max(1.0, abs(x_curr))
        fx_plus = _safe_f(x_curr + h)
        if fx_plus is None:
            break

        d = (fx_plus - f_curr) / h
        if d == 0 or not math.isfinite(d):
            h = 1e-9 * max(1.0, abs(x_curr))
            fx_plus = _safe_f(x_curr + h)
            if fx_plus is None:
                break
            d = (fx_plus - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damp large steps
        max_step = 5.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            break

        if abs(f_new) < abs(f_curr):
            x_curr, f_curr = x_new, f_new
        else:
            break

    # Final polish if we have budget
    if call_count < MAX_CALLS - 5 and abs(f_curr) > TOL:
        # Try a few more secant-like steps
        x_prev = x_curr
        f_prev = f_curr
        for _ in range(POLISH_BUDGET):
            if not _check_time() or call_count >= MAX_CALLS:
                break
            if abs(f_curr) <= TOL:
                break
            h = 1e-7 * max(1.0, abs(x_curr))
            fx_p = _safe_f(x_curr + h)
            if fx_p is None:
                break
            d = (fx_p - f_curr) / h
            if d != 0 and math.isfinite(d):
                x_new = x_curr - f_curr / d
                if math.isfinite(x_new):
                    f_new = _safe_f(x_new)
                    if f_new is not None:
                        if abs(f_new) <= TOL:
                            break
                        if abs(f_new) < abs(f_curr):
                            x_prev, f_prev = x_curr, f_curr
                            x_curr, f_curr = x_new, f_new
                        else:
                            break

    return best_x
# EVOLVE-BLOCK-END
