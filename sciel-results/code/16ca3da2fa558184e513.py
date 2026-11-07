"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust scalar root finder for smooth polynomial equations.
    Uses Regula Falsi with Brent's safeguard (inverse interpolation) to
    achieve superlinear convergence without derivative calls, minimizing
    function evaluations while maintaining bracket guarantees.
    """
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin before 120s limit
    TOL = 1e-9  # internal tolerance, stricter than required 1e-8
    MAX_CALLS = 495  # stay under 500 per equation

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

    # Phase 1: Find a bracket [a, b] with sign change, using few evaluations.
    a, b, fa, fb = None, None, None, None

    # Try expanding steps to find a sign change
    for s in [0.1, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0]:
        if not _check_time() or call_count >= MAX_CALLS - 10:
            break

        # Try right
        xr = x + s
        fr = _safe_f(xr)
        if fr is None:
            continue
        if abs(fr) <= TOL:
            return xr
        if fx is not None and fx * fr < 0:
            a, b, fa, fb = x, xr, fx, fr
            break

        # Try left
        xl = x - s
        fl = _safe_f(xl)
        if fl is None:
            continue
        if abs(fl) <= TOL:
            return xl
        if fx is not None and fx * fl < 0:
            a, b, fa, fb = xl, x, fl, fx
            break

    # If no bracket found, try systematic expansion
    if a is None:
        step = 1.0
        for _ in range(20):
            if not _check_time() or call_count >= MAX_CALLS - 10:
                break

            xr = x + step
            fr = _safe_f(xr)
            if fr is None:
                step *= 2.0
                continue
            if abs(fr) <= TOL:
                return xr
            if fx is not None and fx * fr < 0:
                a, b, fa, fb = x, xr, fx, fr
                break

            xl = x - step
            fl = _safe_f(xl)
            if fl is None:
                step *= 2.0
                continue
            if abs(fl) <= TOL:
                return xl
            if fx is not None and fx * fl < 0:
                a, b, fa, fb = xl, x, fl, fx
                break

            step *= 2.0
            if step > 1e8:
                break

    # Phase 2: If bracket found, use Regula Falsi with Brent's safeguard
    if a is not None and b is not None and fa is not None and fb is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        x_lo, x_hi = a, b
        f_lo, f_hi = fa, fb

        # Track previous point for inverse quadratic interpolation (Brent's method)
        x_prev = None
        f_prev = None

        for _ in range(50):
            if not _check_time() or call_count >= MAX_CALLS - 5:
                break

            if abs(f_lo) <= TOL:
                return x_lo
            if abs(f_hi) <= TOL:
                return x_hi

            # Try inverse quadratic interpolation if we have 3 points
            x_new = None

            if x_prev is not None and f_prev is not None:
                # Inverse quadratic interpolation using (x_lo, f_lo), (x_hi, f_hi), (x_prev, f_prev)
                # Formula: x_new = x_lo + (x_hi - x_lo) * f_lo * (f_hi - f_prev) / ((f_lo - f_hi) * (f_lo - f_prev))
                denom1 = f_lo - f_hi
                denom2 = f_lo - f_prev
                if abs(denom1) > 1e-300 and abs(denom2) > 1e-300:
                    x_iqp = x_lo + (x_hi - x_lo) * f_lo * (f_hi - f_prev) / (denom1 * denom2)
                    # Check if x_iqp is within bracket and not too close to endpoints
                    if x_lo < x_iqp < x_hi:
                        # Ensure it's not too close to x_lo or x_hi (Brent's safeguard)
                        if abs(x_iqp - x_lo) > 1e-10 * abs(x_hi - x_lo) and abs(x_iqp - x_hi) > 1e-10 * abs(x_hi - x_lo):
                            x_new = x_iqp

            # If IQP didn't give a valid point, try secant (Regula Falsi)
            if x_new is None:
                if f_lo != f_hi:
                    x_sec = x_lo - f_lo * (x_lo - x_hi) / (f_lo - f_hi)
                    if x_lo < x_sec < x_hi:
                        x_new = x_sec

            # Fallback to bisection if neither IQP nor secant works
            if x_new is None:
                x_new = (x_lo + x_hi) / 2.0

            # Evaluate at x_new
            f_new = _safe_f(x_new)
            if f_new is None:
                break
            if abs(f_new) <= TOL:
                return x_new

            # Update bracket
            if f_new * f_lo < 0:
                # Root is between x_lo and x_new
                x_prev, f_prev = x_hi, f_hi
                x_hi, f_hi = x_new, f_new
            else:
                # Root is between x_new and x_hi
                x_prev, f_prev = x_lo, f_lo
                x_lo, f_lo = x_new, f_new

        # Return the best endpoint
        if abs(f_lo) < abs(f_hi):
            return x_lo
        else:
            return x_hi

    # Phase 3: No bracket found. Use Newton's method with finite difference.
    x_curr = float(x0)
    f_curr = fx

    for _ in range(30):
        if not _check_time() or call_count >= MAX_CALLS - 5:
            break

        if abs(f_curr) <= TOL:
            return x_curr

        # Finite difference derivative with adaptive step
        h = 1e-8 * max(1.0, abs(x_curr))
        fx_plus = _safe_f(x_curr + h)
        if fx_plus is None:
            break

        d = (fx_plus - f_curr) / h
        if d == 0 or not math.isfinite(d):
            # Try smaller step
            h = 1e-10 * max(1.0, abs(x_curr))
            fx_plus = _safe_f(x_curr + h)
            if fx_plus is None:
                break
            d = (fx_plus - f_curr) / h
            if d == 0 or not math.isfinite(d):
                break

        x_new = x_curr - f_curr / d
        if not math.isfinite(x_new):
            break

        # Damping: limit step to avoid wild overshoots
        max_step = 2.0 * max(1.0, abs(x_curr))
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

        f_new = _safe_f(x_new)
        if f_new is None:
            break

        if abs(f_new) <= TOL:
            return x_new

        # If we're making progress, continue; otherwise try bracketing
        if abs(f_new) < abs(f_curr) * 0.9:
            x_curr = x_new
            f_curr = f_new
        else:
            # Stagnation: try to bracket using current and previous
            if abs(f_new) < abs(f_curr):
                x_curr, f_curr = x_new, f_new
            else:
                # No progress; break and use best found
                break

    # Final: return the best point we've seen
    return best_x
# EVOLVE-BLOCK-END
