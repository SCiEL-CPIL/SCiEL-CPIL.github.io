import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 495
    DEADLINE = time.time() + 145.0  # Safety margin within 150s

    calls = 0
    best_x = float(x0)
    best_abs = float('inf')

    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')

    x = float(x0)
    fx = safe_f(x)
    
    if abs(fx) <= TOL:
        return x

    # Try to find a bracket using expanding intervals
    bracket_a = None
    bracket_b = None
    fa = fx
    fb = None

    # First, check if we can bracket by expanding from x0
    step = max(1.0, abs(x0))
    for i in range(1, 50):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        # Try x0 + step
        x_plus = x0 + step
        fp = safe_f(x_plus)
        if math.isfinite(fp) and fa * fp < 0:
            bracket_a = x
            bracket_b = x_plus
            fb = fp
            break
        # Try x0 - step
        x_minus = x0 - step
        fm = safe_f(x_minus)
        if math.isfinite(fm) and fa * fm < 0:
            bracket_a = x_minus
            bracket_b = x
            fb = fm
            break
        step *= 2.0

    if bracket_a is not None:
        # Use bisection with interpolation (Illinois method or just bisection)
        a = bracket_a
        b = bracket_b
        fa_val = fa
        fb_val = fb
        
        for _ in range(100):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            if abs(b - a) < 1e-15:
                break
            # Try inverse quadratic interpolation if possible, else bisection
            c = (a + b) / 2.0
            fc = safe_f(c)
            if not math.isfinite(fc):
                break
            if abs(fc) <= TOL:
                return c
            if fa_val * fc < 0:
                b = c
                fb_val = fc
            else:
                a = c
                fa_val = fc
            if abs(b - a) < 1e-15:
                break
        
        # Return the midpoint or the one with smaller |f|
        if abs(fa_val) <= abs(fb_val):
            return a
        else:
            return b
    else:
        # No bracket found. Try Newton's method with damping.
        x = float(x0)
        fx = safe_f(x)
        for _ in range(50):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            if abs(fx) <= TOL:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (safe_f(x + h) - fx) / h
            if not math.isfinite(d) or d == 0:
                break
            dx = fx / d
            # Damp if step is too large
            if abs(dx) > abs(x) * 0.5:
                dx = math.copysign(min(abs(dx), abs(x) * 0.5), dx)
            x_new = x - dx
            if not math.isfinite(x_new):
                break
            x = x_new
            fx = safe_f(x)
        return best_x

    return best_x
