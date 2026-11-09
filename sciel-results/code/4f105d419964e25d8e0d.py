"""
Structural change: Brent's method (bisection + secant + inverse quadratic interpolation)
with a compact bracketing scan. Brent's method is the standard robust root-finder and
typically converges in fewer iterations than Illinois for smooth functions, while still
guaranteeing convergence. The bracketing scan is simplified to a single adaptive outward
expansion with sign-change detection, which is more efficient than the previous multi-scale
grid scan.
"""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0
    TOL = 1e-12
    HARD_TOL = 1e-8
    MAX_CALLS = 480

    call_count = 0
    best_x = x0
    best_val = float('inf')
    cache = {}

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def get_f(x):
        nonlocal call_count, best_x, best_val
        if x in cache:
            return cache[x]
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                val = float('nan')
            else:
                val = float(val)
        except Exception:
            call_count += 1
            val = float('nan')
        cache[x] = val
        if val != float('nan'):
            av = abs(val)
            if av < best_val:
                best_val = av
                best_x = x
        return val

    # Evaluate start point
    f0 = get_f(x0)
    if f0 != float('nan') and abs(f0) <= HARD_TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with f(a)*f(b) < 0
    # ------------------------------------------------------------------
    a = b = fa = fb = None

    # Adaptive outward expansion: try both directions with increasing step
    # Start with a small step and double it each time
    h = max(1e-3, abs(x0) * 1e-3)
    
    # Track the most recent point on each side
    left_x, left_f = x0, f0
    right_x, right_f = x0, f0

    for i in range(1, 60):
        if not elapsed_ok() or call_count >= MAX_CALLS - 15:
            break

        # Right side
        x_r = x0 + i * h
        f_r = get_f(x_r)
        if f_r != float('nan'):
            if abs(f_r) <= HARD_TOL:
                return x_r
            if right_f != float('nan') and right_f * f_r < 0:
                a, b = right_x, x_r
                fa, fb = right_f, f_r
                break
            right_x, right_f = x_r, f_r

        # Left side
        x_l = x0 - i * h
        f_l = get_f(x_l)
        if f_l != float('nan'):
            if abs(f_l) <= HARD_TOL:
                return x_l
            if left_f != float('nan') and left_f * f_l < 0:
                a, b = x_l, left_x
                fa, fb = f_l, left_f
                break
            left_x, left_f = x_l, f_l

        # Double the step size
        h *= 2.0

    # If no bracket found via simple expansion, try a denser local scan
    if a is None:
        # Scan [x0 - 1, x0 + 1] with 41 points (step 0.05)
        step = 0.05
        prev_x, prev_f = None, None
        for k in range(-20, 21):
            if not elapsed_ok() or call_count >= MAX_CALLS - 10:
                break
            x_t = x0 + k * step
            f_t = get_f(x_t)
            if f_t != float('nan'):
                if abs(f_t) <= HARD_TOL:
                    return x_t
                if prev_f is not None and prev_f * f_t < 0:
                    a, b = prev_x, x_t
                    fa, fb = prev_f, f_t
                    break
                prev_x, prev_f = x_t, f_t
        if a is not None:
            break

    # ------------------------------------------------------------------
    # 2) Brent's method to refine the root
    # ------------------------------------------------------------------
    if a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent's method implementation
        # Variables: a, b, c with f(a)*f(b) < 0, and |f(b)| <= |f(c)|
        # d, e: previous step and safeguard
        x = b
        c = a
        fc = fa
        fa_val = fa
        fb_val = fb
        d = e = 0.0

        # Ensure |fb| <= |fc|
        if abs(fb_val) > abs(fc):
            a, c = c, a
            fa_val, fc = fc, fa_val
            x = b

        for _ in range(200):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break

            if abs(fb_val) <= TOL:
                return x
            if abs(fc) <= TOL:
                return c

            # Ensure |fb| <= |fc|
            if abs(fb_val) > abs(fc):
                a, c = c, a
                fa_val, fc = fc, fa_val
                x = b

            # Convergence check
            if abs(x - a) < 1e-14 * max(1.0, abs(x)):
                return x if abs(fb_val) < abs(fc) else c

            # Inverse quadratic interpolation
            try:
                # If f(a) == f(c), use secant
                if fa_val == fc:
                    s = fb_val / fa_val
                    # Secant: x_new = a - fa*(a-b)/(fa-fb) but we want to be careful
                    # Standard secant: x_new = (a*fb - b*fa) / (fb - fa)
                    denom = fb_val - fa_val
                    if abs(denom) > 1e-300:
                        x_new = (a * fb_val - b * fa_val) / denom
                    else:
                        x_new = 0.5 * (a + b)
                else:
                    # Inverse quadratic interpolation
                    # x_new = a - fa * ((a-b)*((a-c)*(fa-fb) - (b-c)*(fa-fc))) /
                    #         ((a-c)*(fb-fa) - (a-b)*(fc-fa))
                    num = (a - b) * ((a - c) * (fa_val - fb_val) - (b - c) * (fa_val - fc))
                    den = (a - c) * (fb_val - fa_val) - (a - b) * (fc - fa_val)
                    if abs(den) > 1e-300:
                        x_new = a - fa_val * num / den
                    else:
                        x_new = 0.5 * (a + b)
            except (ZeroDivisionError, OverflowError, FloatingPointError):
                x_new = 0.5 * (a + b)

            # Safeguard: if x_new is not in (min(a,b), max(a,b)), use bisection
            lo = min(a, b)
            hi = max(a, b)
            if not math.isfinite(x_new) or x_new <= lo or x_new >= hi:
                x_new = 0.5 * (a + b)

            # Evaluate f(x_new)
            f_new = get_f(x_new)
            if f_new == float('nan'):
                # Fallback: just bisect
                x_new = 0.5 * (a + b)
                f_new = get_f(x_new)
                if f_new == float('nan'):
                    break

            if abs(f_new) <= TOL:
                return x_new

            # Update points
            e = d
            d = x - x_new

            if fa_val * f_new < 0:
                # Root is between a and x_new
                c = b
                fc = fb_val
                b = x_new
                fb_val = f_new
            else:
                # Root is between x_new and b
                if fb_val * f_new < 0:
                    c = a
                    fc = fa_val
                else:
                    fc = f_new
                a = x_new
                fa_val = f_new

            x = b

    # ------------------------------------------------------------------
    # 3) Return best found
    # ------------------------------------------------------------------
    if math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
