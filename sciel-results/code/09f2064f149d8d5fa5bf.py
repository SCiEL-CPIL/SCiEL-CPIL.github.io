"""Root finder for 3 fixed oscillatory equations."""
import math
import time


def find_root(f, x0):
    """
    Find a root of f starting near x0, using a robust multi-strategy approach.
    Uses adaptive scanning to find brackets, then Brent's method for convergence.
    """
    deadline = time.time() + 140.0
    tol = 1e-9

    x = float(x0)

    # Track best solution
    best_x = x
    best_abs = float('inf')

    def update_best(x_val, f_val):
        nonlocal best_x, best_abs
        if f_val is None or not math.isfinite(x_val) or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_abs = abs_f

    def check_deadline():
        return time.time() >= deadline

    # Cache for f evaluations to avoid redundant calls
    cache = {}

    def f_cached(x_val):
        key = x_val
        if key in cache:
            return cache[key]
        try:
            val = f(x_val)
            if val is None or not math.isfinite(val):
                cache[key] = None
                return None
            cache[key] = val
            return val
        except:
            cache[key] = None
            return None

    # Initial evaluation
    f_x0 = f_cached(x)
    if f_x0 is not None:
        update_best(x, f_x0)
        if abs(f_x0) <= tol:
            return x

    # Phase 1: Adaptive multi-scale scanning to find a bracket
    # Try multiple scales: very fine to coarse
    a = x
    fa = f_x0
    b = None
    fb = None

    if fa is not None and fa != 0:
        # Try a sequence of step sizes, starting fine and going coarse
        # For oscillatory functions, the period might be small or large
        base_steps = [
            1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
            1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0,
            1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0
        ]

        for step in base_steps:
            if check_deadline():
                break

            # Try a few points at this scale in both directions
            for mult in [1, 2, 3, 5]:
                if check_deadline():
                    break

                # Positive direction
                x_pos = x + step * mult
                f_pos = f_cached(x_pos)
                if f_pos is not None:
                    update_best(x_pos, f_pos)
                    if fa * f_pos < 0:
                        a = x
                        b = x_pos
                        fa = f_x0
                        fb = f_pos
                        break

                # Negative direction
                x_neg = x - step * mult
                f_neg = f_cached(x_neg)
                if f_neg is not None:
                    update_best(x_neg, f_neg)
                    if fa * f_neg < 0:
                        a = x_neg
                        b = x
                        fa = f_neg
                        fb = f_x0
                        break

            if b is not None:
                break

    # Phase 2: If bracket found, use Brent's method
    if b is not None and fa is not None and fb is not None and fa * fb < 0:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Brent's method implementation
        # Use the standard Brent algorithm
        c = a
        fc = fa
        d = 0.0
        e = 0.0

        for _ in range(200):
            if check_deadline():
                break

            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b

            # Check for convergence
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                break

            # Try inverse quadratic interpolation or secant
            x_new = None
            secant_ok = False

            # Secant step
            if fa != fb:
                x_sec = b - fb * (b - a) / (fb - fa)
                # Check if secant point is within a reasonable range
                # and not too close to endpoints
                if a < x_sec < b:
                    if abs(x_sec - b) > 1e-14 * max(1.0, abs(b)):
                        x_new = x_sec
                        secant_ok = True

            # If secant not usable, use bisection
            if not secant_ok:
                x_new = (a + b) / 2.0

            f_new = f_cached(x_new)
            if f_new is None:
                break

            update_best(x_new, f_new)

            if abs(f_new) <= tol:
                return x_new

            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new

    # Phase 3: If no bracket found, try local refinement with Newton-like steps
    if b is None:
        # Try a few Newton-like iterations with finite difference
        x_newton = x
        for _ in range(30):
            if check_deadline():
                break
            fx = f_cached(x_newton)
            if fx is None:
                break
            update_best(x_newton, fx)
            if abs(fx) <= tol:
                return x_newton
            h = 1e-5 * max(1.0, abs(x_newton))
            f_plus = f_cached(x_newton + h)
            if f_plus is None:
                break
            d = (f_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_candidate = x_newton - fx / d
            if not math.isfinite(x_candidate):
                break
            x_newton = x_candidate

    # Phase 4: Try secant method with two nearby points
    if b is None:
        x0_sec = x
        x1_sec = x + max(1e-4, 1e-4 * abs(x))
        fx0 = f_cached(x0_sec)
        fx1 = f_cached(x1_sec)

        if fx0 is not None and fx1 is not None:
            for _ in range(50):
                if check_deadline():
                    break
                if abs(fx0) <= tol:
                    return x0_sec
                if abs(fx1) <= tol:
                    return x1_sec
                if fx0 == fx1:
                    break
                x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
                if not math.isfinite(x2_sec):
                    break
                fx2 = f_cached(x2_sec)
                if fx2 is None:
                    break
                update_best(x2_sec, fx2)
                x0_sec = x1_sec
                fx0 = fx1
                x1_sec = x2_sec
                fx1 = fx2

    # Return the best solution found
    if best_abs <= 1e-8:
        return best_x
    return best_x
