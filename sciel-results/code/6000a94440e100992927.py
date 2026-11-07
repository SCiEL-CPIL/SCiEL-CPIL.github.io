"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x = float(x0)
    deadline = time.monotonic() + 118.0
    best_x = x
    best_fx = None
    call_count = 0

    def safe_f(val):
        nonlocal call_count
        try:
            result = f(val)
            call_count += 1
            if result is not None and math.isfinite(result):
                return float(result)
        except Exception:
            pass
        return None

    def is_success(fx_val):
        return fx_val is not None and abs(fx_val) <= 1e-8

    def time_ok():
        return time.monotonic() < deadline

    # Evaluate at starting point
    best_fx = safe_f(x)
    if is_success(best_fx):
        return x

    # --- Phase 1: Try to find a bracket around x0 ---
    # For polynomials, a sign change is reliable. Try multiple scales.
    scale = max(1.0, abs(x0))
    left = right = None
    f_left = f_right = None

    for factor in [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0]:
        if not time_ok():
            break
        lo = x0 - scale * factor
        hi = x0 + scale * factor
        f_lo = safe_f(lo)
        f_hi = safe_f(hi)
        if f_lo is None or f_hi is None:
            continue
        if f_lo * f_hi <= 0:
            left, right = lo, hi
            f_left, f_right = f_lo, f_hi
            break
        # Also check if either endpoint is already good
        if is_success(f_lo):
            return lo
        if is_success(f_hi):
            return hi
        if best_fx is None or abs(f_lo) < abs(best_fx):
            best_x, best_fx = lo, f_lo
        if best_fx is None or abs(f_hi) < abs(best_fx):
            best_x, best_fx = hi, f_hi

    # --- Phase 2: Brent-style hybrid root finding with bracket ---
    if left is not None and right is not None and f_left * f_right <= 0:
        # We have a bracket. Use a hybrid method:
        # - Try secant/IQI steps
        # - Fall back to bisection if step leaves bracket
        a, b = left, right
        fa, fb = f_left, f_right
        c = a
        fc = fa
        # Ensure |f(a)| <= |f(b)|
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        c = b
        fc = fb

        for _ in range(100):
            if not time_ok():
                break
            if abs(fb) <= 1e-10:
                return b
            if is_success(fb):
                return b
            if best_fx is None or abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

            # Bisection
            d = (a - b) / 2.0
            e = fa - fb

            # Try secant or IQI
            use_secant = True
            if abs(e) < abs(d):
                # Try inverse quadratic interpolation
                # Need three points: we have (a,fa), (b,fb), (c,fc)
                # IQI: x = b - fb * ((b-a)*(fb-fc) - (b-c)*(fb-fa)) / ((fa-fc)*(fa-fb))
                denom = (fa - fc) * (fa - fb)
                if denom != 0 and math.isfinite(denom):
                    num = (b - a) * (fb - fc) - (b - c) * (fb - fa)
                    x_iqi = b - fb * num / denom
                    # Check if x_iqi is within bracket
                    if min(a, b) < x_iqi < max(a, b):
                        d = x_iqi - b
                        e = d
                        use_secant = False
                    else:
                        use_secant = True
                else:
                    use_secant = True

            if use_secant:
                # Secant method
                if fa != fb:
                    x_sec = b - fb * (b - a) / (fb - fa)
                    if min(a, b) < x_sec < max(a, b):
                        d = x_sec - b
                        e = d
                    else:
                        # Fall back to bisection
                        d = (a - b) / 2.0
                        e = d
                else:
                    d = (a - b) / 2.0
                    e = d

            # If step too small, use bisection
            if abs(d) < 1e-15 * max(1.0, abs(b)):
                d = (a - b) / 2.0
                e = d

            # Update
            x_new = b + e
            fx_new = safe_f(x_new)
            if fx_new is None:
                # Fall back to bisection
                x_new = (a + b) / 2.0
                fx_new = safe_f(x_new)
                if fx_new is None:
                    break

            c, fc = b, fb
            b, fb = x_new, fx_new

            if is_success(fb):
                return b
            if best_fx is None or abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

            # Maintain bracket
            if fa * fb <= 0:
                a, fa = c, fc
            else:
                a, fa = b, fb

            # Check convergence
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break

        if is_success(fb):
            return b
        if best_fx is not None and abs(best_fx) <= 1e-8:
            return best_x

    # --- Phase 3: If no bracket found, try Newton with damping ---
    x_newton = x
    fx = best_fx
    for _ in range(40):
        if not time_ok():
            break
        if fx is None:
            break
        if is_success(fx):
            return x_newton

        h = 1e-6 * max(1.0, abs(x_newton))
        fx_plus = safe_f(x_newton + h)
        if fx_plus is None:
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break

        step = fx / d
        if abs(step) > 10 * max(1.0, abs(x_newton)):
            step = math.copysign(10 * max(1.0, abs(x_newton)), step)

        x_newton = x_newton - step
        if not math.isfinite(x_newton):
            break

        fx = safe_f(x_newton)
        if fx is None:
            break

        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x_newton
            best_fx = fx

        if is_success(fx):
            return x_newton

    if is_success(best_fx):
        return best_x

    # --- Phase 4: Secant method without bracket ---
    scale = max(1.0, abs(x0))
    x1 = x0 + scale * 0.1
    fx1 = safe_f(x1)
    if fx1 is not None and best_fx is not None:
        x0_sec = x
        x1_sec = x1
        fx0 = best_fx
        fx1_sec = fx1

        for _ in range(30):
            if not time_ok():
                break
            if fx1_sec is None or fx0 is None:
                break
            if is_success(fx1_sec):
                return x1_sec

            denom = fx1_sec - fx0
            if denom == 0 or not math.isfinite(denom):
                break

            x2 = x1_sec - fx1_sec * (x1_sec - x0_sec) / denom
            if not math.isfinite(x2):
                break

            fx2 = safe_f(x2)
            if fx2 is None:
                break

            if best_fx is None or abs(fx2) < abs(best_fx):
                best_x = x2
                best_fx = fx2

            if is_success(fx2):
                return x2

            x0_sec = x1_sec
            fx0 = fx1_sec
            x1_sec = x2
            fx1_sec = fx2

    if is_success(best_fx):
        return best_x

    # --- Phase 5: Last resort - try more points ---
    for offset in [0.01, -0.01, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
        if not time_ok():
            break
        x_try = x0 + offset * scale
        fx_try = safe_f(x_try)
        if fx_try is not None:
            if best_fx is None or abs(fx_try) < abs(best_fx):
                best_x = x_try
                best_fx = fx_try
            if is_success(fx_try):
                return x_try

    return best_x
# EVOLVE-BLOCK-END
