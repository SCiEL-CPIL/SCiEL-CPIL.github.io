import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0
    max_calls = 480
    call_count = 0

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    x = float(x0)
    if not math.isfinite(x):
        return x

    fx = f_call(x)

    if not math.isfinite(fx):
        return x

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_f = fx

    # Phase 1: Find a bracket by geometric expansion
    a = x
    fa = fx
    b = None
    fb = None
    bracket_found = False

    # Try expanding right
    h = 0.5
    for _ in range(25):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if abs(fb) < abs(best_f):
            best_x = b
            best_f = fb
        if fa * fb < 0:
            bracket_found = True
            break
        a = b
        fa = fb
        h *= 2.0

    if not bracket_found:
        # Try expanding left from original x0
        a = x
        fa = fx
        b = None
        fb = None
        h = 0.5
        for _ in range(25):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if abs(fb) < abs(best_f):
                best_x = b
                best_f = fb
            if fa * fb < 0:
                bracket_found = True
                break
            a = b
            fa = fb
            h *= 2.0

    if bracket_found:
        # We have a bracket [a, b] with fa * fb < 0
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Secant-first hybrid: try secant step, fall back to bisection
        # This converges faster than pure bisection for smooth polynomials
        for _ in range(150):
            if not time_ok():
                break

            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b

            # Check convergence by bracket width
            if abs(b - a) < 1e-13 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b

            # Try secant step
            if fb != fa:
                x_secant = b - fb * (b - a) / (fb - fa)
            else:
                x_secant = 0.5 * (a + b)

            # Use secant if it's inside the bracket, otherwise bisect
            if a < x_secant < b:
                s = x_secant
            else:
                s = 0.5 * (a + b)

            fs = f_call(s)

            if not math.isfinite(fs):
                s = 0.5 * (a + b)
                fs = f_call(s)

            if abs(fs) <= 1e-8:
                return s
            if abs(fs) < abs(best_f):
                best_x = s
                best_f = fs

            # Update bracket: keep the subinterval with sign change
            if fa * fs < 0:
                b = s
                fb = fs
            else:
                a = s
                fa = fs

        # Return the point with smallest |f|
        return a if abs(fa) < abs(fb) else b
    else:
        # No bracket found. Use Newton's method with finite differences
        # and backtracking line search.
        x_curr = x
        fx_curr = fx
        h_fd = 1e-7 * max(1.0, abs(x_curr))

        for _ in range(60):
            if not time_ok():
                break

            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_curr) < abs(best_f):
                best_x = x_curr
                best_f = fx_curr

            # Finite difference derivative
            fx_plus = f_call(x_curr + h_fd)
            if not math.isfinite(fx_plus):
                fx_plus = f_call(x_curr - h_fd)
            if not math.isfinite(fx_plus):
                break

            d = (fx_plus - fx_curr) / h_fd
            if d == 0 or not math.isfinite(d):
                break

            # Newton step
            x_new = x_curr - fx_curr / d

            if not math.isfinite(x_new):
                break

            # Backtracking: try full step, then halve until improvement
            fx_new = f_call(x_new)

            if not math.isfinite(fx_new):
                break

            if abs(fx_new) < abs(best_f):
                best_x = x_new
                best_f = fx_new

            if abs(fx_new) < abs(fx_curr):
                x_curr = x_new
                fx_curr = fx_new
            else:
                # Try damped steps
                for damp in [0.5, 0.25, 0.125]:
                    if not time_ok():
                        break
                    x_damped = x_curr - damp * fx_curr / d
                    if not math.isfinite(x_damped):
                        continue
                    fx_damped = f_call(x_damped)
                    if not math.isfinite(fx_damped):
                        continue
                    if abs(fx_damped) < abs(best_f):
                        best_x = x_damped
                        best_f = fx_damped
                    if abs(fx_damped) < abs(fx_curr):
                        x_curr = x_damped
                        fx_curr = fx_damped
                        break

        return best_x
