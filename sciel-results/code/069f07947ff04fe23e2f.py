"""Root finder for smooth polynomial equations - secant + IQI hybrid with central-diff polish."""
import math
import time


def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 110.0  # Safety margin before 150s limit

    def elapsed_ok():
        return time.monotonic() < deadline

    x = float(x0)

    # Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Try nearby points to get a finite evaluation
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                xt = x + sign * dx
                try:
                    fxt = f(xt)
                except Exception:
                    continue
                if math.isfinite(fxt):
                    x, fx = xt, fxt
                    break
            if math.isfinite(fx):
                break
        if not math.isfinite(fx):
            return x0

    if abs(fx) <= 1e-8:
        return x

    best_x, best_f = x, fx

    # Phase 1: Quick bracket search
    a, b = x, x
    fa, fb = fx, fx
    step = 1.0
    bracketed = False

    for i in range(12):
        if not elapsed_ok():
            break
        a = x - step
        b = x + step
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            step *= 2.0
            continue

        if not (math.isfinite(fa) and math.isfinite(fb)):
            step *= 2.0
            continue

        if abs(fa) < abs(best_f):
            best_x, best_f = a, fa
        if abs(fb) < abs(best_f):
            best_x, best_f = b, fb

        if fa == 0.0:
            return a
        if fb == 0.0:
            return b
        if fa * fb < 0.0:
            bracketed = True
            break
        step *= 2.0

    if bracketed:
        # Use IQI with bisection fallback inside the bracket
        lo, hi = a, b
        flo, fhi = fa, fb
        if abs(flo) > abs(fhi):
            lo, hi = hi, lo
            flo, fhi = fhi, flo

        # Initialize three points for IQI: lo, hi, and midpoint
        x_mid = 0.5 * (lo + hi)
        try:
            f_mid = f(x_mid)
        except Exception:
            f_mid = None

        if f_mid is not None and math.isfinite(f_mid):
            if abs(f_mid) < abs(best_f):
                best_x, best_f = x_mid, f_mid
            if abs(f_mid) <= 1e-8:
                return x_mid
            # Update bracket
            if f_mid * flo < 0:
                hi, fhi = x_mid, f_mid
            else:
                lo, flo = x_mid, f_mid
        else:
            # No third point; start with secant
            x_mid = None
            f_mid = None

        # Main iteration: IQI when 3 points available, else secant
        # Points: (x0, f0), (x1, f1), (x2, f2) where x2 is current best
        if f_mid is not None:
            # We have 3 points: lo, x_mid (now either lo or hi side), hi
            # Let's restructure: keep track of 3 most recent distinct points
            # For simplicity, use secant first then switch to IQI
            x1, f1 = lo, flo
            x2, f2 = hi, fhi
        else:
            x1, f1 = lo, flo
            x2, f2 = hi, fhi

        x_curr = x2
        f_curr = f2

        for it in range(80):
            if not elapsed_ok():
                break

            if abs(f_curr) <= 1e-8:
                return x_curr

            # Secant step using last two points
            if abs(x2 - x1) < 1e-14:
                # Degenerate: bisect
                x_new = 0.5 * (lo + hi)
            else:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)

            # Safeguard: stay within bracket
            if x_new <= lo or x_new >= hi:
                x_new = 0.5 * (lo + hi)

            try:
                f_new = f(x_new)
            except Exception:
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
            if not math.isfinite(f_new):
                x_new = 0.5 * (lo + hi)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= 1e-8:
                return x_new

            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            # Update bracket
            if f_new * flo < 0:
                hi, fhi = x_new, f_new
            else:
                lo, flo = x_new, f_new

            # Shift points for next secant
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
            x_curr = x_new
            f_curr = f_new

        return best_x
    else:
        # No bracket: use secant method with step limiting
        x_curr = x
        f_curr = fx

        # Get a second point for initial secant
        h = 1e-7 * max(1.0, abs(x_curr))
        x2 = x_curr + h
        try:
            f2 = f(x2)
        except Exception:
            f2 = None

        if f2 is not None and math.isfinite(f2):
            if abs(f2) < abs(best_f):
                best_x, best_f = x2, f2
            if abs(f2) <= 1e-8:
                return x2
        else:
            # Try another offset
            h = 1e-5 * max(1.0, abs(x_curr))
            x2 = x_curr - h
            try:
                f2 = f(x2)
            except Exception:
                f2 = None
            if f2 is None or not math.isfinite(f2):
                # Fallback: simple bisection-like search
                for i in range(50):
                    if not elapsed_ok():
                        break
                    h = 1e-4 * (i + 1) * max(1.0, abs(x_curr))
                    for sign in [1, -1]:
                        xt = x_curr + sign * h
                        try:
                            ft = f(xt)
                        except Exception:
                            continue
                        if math.isfinite(ft) and abs(ft) < abs(best_f):
                            best_x, best_f = xt, ft
                            if abs(ft) <= 1e-8:
                                return xt
                return best_x

        x1, f1 = x_curr, f_curr
        x2, f2 = x2, f2

        for it in range(100):
            if not elapsed_ok():
                break

            if abs(f2) <= 1e-8:
                return x2

            # Secant step
            if abs(f2 - f1) < 1e-15:
                # Degenerate: take a small step
                x_new = x2 + 1e-4 * max(1.0, abs(x2))
            else:
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)

            # Safeguard: limit step size
            max_step = max(1.0, abs(x2)) * 10.0
            if abs(x_new - x2) > max_step:
                x_new = x2 + max_step * (1.0 if x_new > x2 else -1.0)

            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                # Try a smaller step
                x_new = 0.5 * (x1 + x2)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

            if abs(f_new) <= 1e-8:
                return x_new

            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

            # Accept if it improves, else dampen
            if abs(f_new) < abs(f2):
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
            else:
                # Dampen: take halfway
                x_half = 0.5 * (x2 + x_new)
                try:
                    f_half = f(x_half)
                except Exception:
                    break
                if not math.isfinite(f_half):
                    break

                if abs(f_half) < abs(f2):
                    x1, f1 = x2, f2
                    x2, f2 = x_half, f_half
                else:
                    # Try quarter
                    x_quarter = 0.75 * x2 + 0.25 * x_new
                    try:
                        f_quarter = f(x_quarter)
                    except Exception:
                        break
                    if not math.isfinite(f_quarter):
                        break

                    if abs(f_quarter) < abs(f2):
                        x1, f1 = x2, f2
                        x2, f2 = x_quarter, f_quarter
                    else:
                        # Stagnation: perturb
                        x_new = x2 + 1e-3 * max(1.0, abs(x2))
                        try:
                            f_new = f(x_new)
                        except Exception:
                            break
                        if not math.isfinite(f_new):
                            break
                        x1, f1 = x2, f2
                        x2, f2 = x_new, f_new

        # Final polish with central-difference Newton
        if abs(best_f) > 1e-8 and abs(best_f) < 1.0:
            x_curr = best_x
            f_curr = best_f
            for it in range(30):
                if not elapsed_ok():
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr

                h = 1e-9 * max(1.0, abs(x_curr))
                try:
                    f_plus = f(x_curr + h)
                    f_minus = f(x_curr - h)
                except Exception:
                    break
                if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
                    break

                d = (f_plus - f_minus) / (2 * h)
                if d == 0 or not math.isfinite(d):
                    break

                x_new = x_curr - f_curr / d

                max_step = max(1.0, abs(x_curr)) * 0.1
                if abs(x_new - x_curr) > max_step:
                    x_new = x_curr + max_step * (1.0 if x_new > x_curr else -1.0)

                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

                if abs(f_new) < abs(f_curr):
                    x_curr, f_curr = x_new, f_new
                    if abs(f_curr) < abs(best_f):
                        best_x, best_f = x_curr, f_curr
                else:
                    break

    return best_x
