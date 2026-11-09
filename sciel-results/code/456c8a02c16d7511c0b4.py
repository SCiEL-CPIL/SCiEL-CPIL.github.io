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

    # Step 1: Find a bracket by expanding in both directions
    a = x
    fa = fx
    b = None
    fb = None

    # Try expanding right first
    h = 1.0
    for _ in range(20):
        if not time_ok():
            break
        b = a + h
        fb = f_call(b)
        if not math.isfinite(fb):
            h *= 2.0
            continue
        if fa * fb < 0:
            break
        a = b
        fa = fb
        h *= 2.0

    # If no bracket found going right, try left
    if b is None or fa * fb >= 0:
        a = x
        fa = fx
        b = None
        fb = None
        h = 1.0
        for _ in range(20):
            if not time_ok():
                break
            b = a - h
            fb = f_call(b)
            if not math.isfinite(fb):
                h *= 2.0
                continue
            if fa * fb < 0:
                break
            a = b
            fa = fb
            h *= 2.0

    if b is not None and fa * fb < 0:
        # We have a bracket [a, b] with f(a)*f(b) < 0
        if a > b:
            a, b = b, a
            fa, fb = fb, fa

        # Use secant-bisection hybrid: secant step if inside bracket, else bisection
        x1 = a
        f1 = fa
        x2 = b
        f2 = fb

        for _ in range(100):
            if not time_ok():
                break
            if abs(f2) <= 1e-8:
                return x2
            if abs(f1) <= 1e-8:
                return x1
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                return 0.5 * (a + b)

            # Secant step
            if f2 != f1:
                x_secant = x2 - f2 * (x2 - x1) / (f2 - f1)
            else:
                x_secant = 0.5 * (a + b)

            if a < x_secant < b:
                x_next = x_secant
            else:
                x_next = 0.5 * (a + b)

            f_next = f_call(x_next)

            if not math.isfinite(f_next):
                x_next = 0.5 * (a + b)
                f_next = f_call(x_next)

            if abs(f_next) <= 1e-8:
                return x_next

            # Update bracket
            if fa * f_next < 0:
                b = x_next
                fb = f_next
                x2 = x_next
                f2 = f_next
            else:
                a = x_next
                fa = f_next
                x1 = x_next
                f1 = f_next

        return x2 if abs(f2) < abs(f1) else x1

    else:
        # No bracket found. Try Newton's method with finite differences.
        x = float(x0)
        fx = f_call(x)
        
        for _ in range(60):
            if not time_ok():
                break
            if abs(fx) <= 1e-8:
                return x
                
            h = 1e-6 * max(1.0, abs(x))
            fx_plus = f_call(x + h)
            d = (fx_plus - fx) / h
            
            if d == 0 or not math.isfinite(d):
                break
                
            x_new = x - fx / d
            
            if not math.isfinite(x_new):
                break
                
            # Damping
            if abs(x_new - x) > 10.0:
                x_new = x + 10.0 * math.copysign(1.0, x_new - x)
                
            x = x_new
            fx = f_call(x)
            
        return x
