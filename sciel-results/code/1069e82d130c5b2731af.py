"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using a combination of
    bisection, secant, and Newton methods with call-budget tracking.
    """
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)

    # Track best solution found so far
    best_x = x
    best_f = f(x)
    call_count = 1

    def safe_eval(x_val):
        nonlocal call_count, best_x, best_f
        if not _is_finite(x_val):
            return None
        fx_val = f(x_val)
        call_count += 1
        if not _is_finite(fx_val):
            return None
        if abs(fx_val) < abs(best_f):
            best_f = fx_val
            best_x = x_val
        return fx_val

    # Initial evaluation
    fx = best_f
    if abs(fx) <= 1e-8:
        return best_x

    # Strategy 1: Try to establish a bracket via expansion
    # First, try a small step in the direction of the derivative estimate
    h0 = 1e-4 * max(1.0, abs(x))
    x2 = x + h0
    fx2 = safe_eval(x2)
    if fx2 is None:
        x2 = x - h0
        fx2 = safe_eval(x2)
        if fx2 is None:
            # Fall back to pure secant from two nearby points
            x2 = x + h0
            fx2 = safe_eval(x2)
            if fx2 is None:
                return best_x

    # Check if we have a sign change
    if fx * fx2 < 0:
        a, b = x, x2
        fa, fb = fx, fx2
        # Bisection to narrow the bracket
        for _ in range(40):
            if call_count >= 480 or _time_left() <= 0:
                break
            m = (a + b) / 2.0
            fm = safe_eval(m)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return m
            if fa * fm < 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        # Use the best point from bisection
        if abs(best_f) <= 1e-8:
            return best_x
        # Refine with secant using the last two bracket points
        x1, x2 = a, b
        f1, f2 = fa, fb
        for _ in range(20):
            if call_count >= 490 or _time_left() <= 0:
                break
            if abs(f2 - f1) < 1e-15:
                break
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not _is_finite(x_new):
                break
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        return best_x
    else:
        # No sign change; use secant/Newton hybrid
        # Estimate derivative via finite difference
        x1, f1 = x, fx
        x2, f2 = x2, fx2
        for _ in range(40):
            if call_count >= 480 or _time_left() <= 0:
                break
            if abs(f2 - f1) < 1e-15:
                # Derivative is nearly zero; try a different step
                h = 1e-4 * max(1.0, abs(x2))
                x3 = x2 + h
                f3 = safe_eval(x3)
                if f3 is None:
                    break
                x1, f1 = x2, f2
                x2, f2 = x3, f3
                continue
            # Secant step
            x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
            if not _is_finite(x_new):
                break
            # Damping if the step is too large
            if abs(x_new - x2) > 10 * max(1.0, abs(x2)):
                x_new = x2 + 0.5 * (x_new - x2)
            f_new = safe_eval(x_new)
            if f_new is None:
                break
            if abs(f_new) <= 1e-8:
                return x_new
            # Check for stagnation
            if abs(f_new - f2) < 1e-15 and abs(x_new - x2) < 1e-12:
                break
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new

        # If secant didn't converge, try a Newton refinement with FD derivative
        if abs(best_f) > 1e-8:
            x_n = best_x
            for _ in range(15):
                if call_count >= 495 or _time_left() <= 0:
                    break
                h = 1e-6 * max(1.0, abs(x_n))
                fx_n = safe_eval(x_n)
                if fx_n is None:
                    break
                fx_nh = safe_eval(x_n + h)
                if fx_nh is None:
                    break
                d = (fx_nh - fx_n) / h
                if d == 0:
                    break
                x_new = x_n - fx_n / d
                if not _is_finite(x_new):
                    break
                if abs(x_new - x_n) < 1e-12:
                    break
                x_n = x_new

        return best_x


# EVOLVE-BLOCK-END
