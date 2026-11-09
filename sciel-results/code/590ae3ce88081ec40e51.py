import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using adaptive bracketing and Illinois method.
    Designed for oscillatory functions. Minimizes function calls.
    """
    start_time = time.time()
    deadline = start_time + 10.0

    def time_ok():
        return time.time() < deadline

    x = float(x0)

    # Check initial point
    fx = f(x)
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    # Adaptive bracketing: find [a, b] with sign change
    a = x
    fa = fx
    step = 1e-4 * max(1.0, abs(x))
    b = None
    fb = None

    for direction in [1.0, -1.0]:
        for i in range(20):
            if not time_ok():
                break
            b = x + direction * step
            try:
                fb = f(b)
            except Exception:
                break
            if not math.isfinite(fb):
                break
            if fa * fb < 0:
                break
            step *= 2.0
        if b is not None and fa * fb < 0:
            break

    # If no bracket found, try denser search
    if b is None or fa * fb >= 0:
        window = max(1.0, abs(x)) * 0.1
        n_points = 40
        prev_x = x - window
        try:
            prev_fx = f(prev_x)
        except Exception:
            prev_fx = None
        for i in range(1, n_points + 1):
            if not time_ok():
                break
            cx = x - window + 2 * window * i / n_points
            try:
                cx_f = f(cx)
            except Exception:
                break
            if not math.isfinite(cx_f):
                prev_x, prev_fx = cx, cx_f
                continue
            if prev_fx is not None and math.isfinite(prev_fx) and prev_fx * cx_f < 0:
                a, fa = prev_x, prev_fx
                b, fb = cx, cx_f
                break
            prev_x, prev_fx = cx, cx_f

    if b is None or fa * fb >= 0:
        return x

    # Ensure |fa| < |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    # Illinois method: O(1) extra calls per iteration (reuses halved value)
    max_iters = 300
    for _ in range(max_iters):
        if not time_ok():
            break
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
            break

        # Illinois: shrink the side with larger |f|
        if abs(fa) < abs(fb):
            # Shrink b side
            b = (a + b) / 2.0
            fb = fb / 2.0  # Reuse halved value, no new function call
        else:
            # Shrink a side
            a = (a + b) / 2.0
            fa = fa / 2.0  # Reuse halved value

        # Re-order so |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # Return point with smaller |f|
    if abs(fa) <= abs(fb):
        return a
    return b

# EVOLVE-BLOCK-END
