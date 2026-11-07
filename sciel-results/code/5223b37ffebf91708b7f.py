"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Two-phase approach: fast Newton/secant first, then robust bisection fallback.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit
    TOL = 1e-9  # internal tolerance, slightly stricter than required 1e-8
    MAX_CALLS = 495  # stay under 500 per equation

    x = float(x0)
    call_count = 0
    best_x = x
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

    # Phase 0: Evaluate at x0
    fx = _safe_f(x)
    if fx is None:
        return best_x
    if abs(fx) <= TOL:
        return x

    # Phase 1: Fast Newton/secant with finite differences and call reuse
    # For smooth polynomials, this should converge in very few iterations
    x_curr = x
    f_curr = fx

    # Try Newton's method with forward finite difference
    # Reuse the current f evaluation
    converged = False
    for _ in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        # Forward difference for derivative estimate
        h = 1e-7 * max(1.0, abs(x_curr))
        x_plus = x_curr + h
        f_plus = _safe_f(x_plus)
        if f_plus is None:
            break

        d = (f_plus - f_curr) / h
        if not math.isfinite(d) or abs(d) < 1e-15:
            # Try smaller step
            h = 1e-9 * max(1.0, abs(x_curr))
            x_plus2 = x_curr + h
            f_plus2 = _safe_f(x_plus2)
            if f_plus2 is None:
                break
            d = (f_plus2 - f_curr) / h
            if not math.isfinite(d) or abs(d) < 1e-15:
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damping: limit step size
        max_step = 2.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1 if x_new > x_curr else -1)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # If the step made things worse, try secant instead
        if abs(f_new) > 2.0 * abs(f_curr):
            # Switch to secant with the two best points
            break

        x_curr = x_new
        f_curr = f_new

    # If Newton didn't converge quickly, try secant with bracketing
    # First, try to find a bracket
    a, b = None, None
    fa, fb = None, None

    # Try to find a bracket by expanding from x0
    step = max(0.1, abs(x0) * 0.1)
    fx0 = fx  # reuse

    for _ in range(20):
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        # Try right
        xr = x0 + step
        fr = _safe_f(xr)
        if fr is not None:
            if abs(fr) <= TOL:
                return xr
            if fx0 * fr < 0:
                a, b = x0, xr
                fa, fb = fx0, fr
                break

        # Try left
        xl = x0 - step
        fl = _safe_f(xl)
        if fl is not None:
            if abs(fl) <= TOL:
                return xl
            if fx0 * fl < 0:
                a, b = xl, x0
                fa, fb = fl, fx0
                break

        step *= 2.0
        if step > 1e8:
            break

    if a is not None and b is not None and fa is not None and fb is not None:
        # Hybrid bisection + secant within bracket
        if fa * fb > 0:
            a, b, fa, fb = b, a, fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Keep two most recent points for secant
        x_prev, f_prev = x_hi, f_hi
        x_curr2, f_curr2 = x_lo, f_lo

        for _ in range(60):
            if not _check_time() or call_count >= MAX_CALLS - 3:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Try secant first if we have two distinct points with different f values
            used_secant = False
            if x_prev != x_curr2 and f_prev != f_curr2:
                x_sec = x_curr2 - f_curr2 * (x_curr2 - x_prev) / (f_curr2 - f_prev)
                if x_lo + 1e-15 < x_sec < x_hi - 1e-15:
                    f_sec = _safe_f(x_sec)
                    if f_sec is not None:
                        if abs(f_sec) <= TOL:
                            return x_sec
                        if f_sec * f_lo < 0:
                            x_hi = x_sec
                            f_hi = f_sec
                        else:
                            x_lo = x_sec
                            f_lo = f_sec
                        x_prev, f_prev = x_curr2, f_curr2
                        x_curr2, f_curr2 = x_sec, f_sec
                        used_secant = True

            if not used_secant:
                # Bisection
                x_mid = (x_lo + x_hi) / 2.0
                f_mid = _safe_f(x_mid)
                if f_mid is None:
                    break
                if abs(f_mid) <= TOL:
                    return x_mid

                if f_mid * f_lo < 0:
                    x_hi = x_mid
                    f_hi = f_mid
                else:
                    x_lo = x_mid
                    f_lo = f_mid

                x_prev, f_prev = x_curr2, f_curr2
                x_curr2, f_curr2 = x_mid, f_mid

        if abs(f_lo) <= abs(f_hi):
            return x_lo
        else:
            return x_hi

    # No bracket found: fall back to damped secant
    x_curr = x
    f_curr = fx
    h = 0.1 * max(1.0, abs(x))
    x_prev = x + h
    f_prev = _safe_f(x_prev)

    if f_prev is None:
        return best_x

    for _ in range(40):
        if not _check_time() or call_count >= MAX_CALLS - 3:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        if f_prev != f_curr and x_prev != x_curr:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        else:
            x_new = x_curr + h
            h *= 0.5

        if not math.isfinite(x_new):
            break

        max_step = 2.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1 if x_new > x_curr else -1)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        if abs(f_new) < abs(f_curr):
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_new, f_new
        else:
            x_prev, f_prev = x_curr, f_curr

    return best_x
# EVOLVE-BLOCK-END
