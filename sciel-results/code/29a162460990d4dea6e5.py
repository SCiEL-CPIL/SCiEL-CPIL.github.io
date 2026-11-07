import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8, using a local Taylor model
    built from a few initial samples, then verify with actual calls.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    x0 = float(x0)
    # First, evaluate f at x0
    try:
        fx0 = f(x0)
    except Exception:
        return x0

    if not math.isfinite(fx0):
        # If f(x0) is non-finite, we can't do much. Return x0 as best guess.
        return x0

    if abs(fx0) <= 1e-8:
        return x0

    # Step 1: Build a local Taylor polynomial using finite differences.
    # We'll use a 5-point stencil to get derivatives up to order 4.
    # Choose a step size h based on x0 scale.
    scale = max(1.0, abs(x0))
    h = 1e-4 * scale

    # Sample points: x0 - 2h, x0 - h, x0, x0 + h, x0 + 2h
    # We already have f(x0). Need 4 more calls.
    points = [x0 - 2*h, x0 - h, x0, x0 + h, x0 + 2*h]
    values = [None] * 5
    values[2] = fx0  # already computed

    # Evaluate the other 4 points
    for i in [0, 1, 3, 4]:
        if time.time() > deadline:
            break
        try:
            values[i] = f(points[i])
        except Exception:
            values[i] = None

    # Check if any of these are roots
    for i in [0, 1, 3, 4]:
        if values[i] is not None:
            if math.isfinite(values[i]) and abs(values[i]) <= 1e-8:
                return points[i]
            if not math.isfinite(values[i]):
                values[i] = None  # mark as invalid

    # If we have enough valid values, build the Taylor polynomial.
    # Coefficients of the polynomial in terms of (x - x0)/h = t
    # f(x0 + t*h) ≈ c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
    # Using 5-point stencil formulas:
    # c0 = f(x0)
    # c1 = (-f(x0-2h) + 8f(x0-h) - 8f(x0+h) + f(x0+2h)) / (12h)
    # c2 = (f(x0-2h) - 16f(x0-h) + 30f(x0) - 16f(x0+h) + f(x0+2h)) / (12h^2)
    # c3 = (-f(x0-2h) + 2f(x0-h) - 2f(x0+h) + f(x0+2h)) / (2h^3)
    # c4 = (f(x0-2h) - 4f(x0-h) + 6f(x0) - 4f(x0+h) + f(x0+2h)) / (h^4)

    valid = all(v is not None for v in values)
    if valid:
        v0, v1, v2, v3, v4 = values
        h2 = h * h
        h3 = h2 * h
        h4 = h2 * h2

        c0 = v2
        c1 = (-v0 + 8*v1 - 8*v3 + v4) / (12 * h)
        c2 = (v0 - 16*v1 + 30*v2 - 16*v3 + v4) / (12 * h2)
        c3 = (-v0 + 2*v1 - 2*v3 + v4) / (2 * h3)
        c4 = (v0 - 4*v1 + 6*v2 - 4*v3 + v4) / h4

        # Now we have a polynomial P(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4
        # We want to find t such that P(t) = 0.
        # Use Newton's method on P(t).
        # P'(t) = c1 + 2*c2*t + 3*c3*t^2 + 4*c4*t^3

        def P(t):
            return c0 + c1*t + c2*t*t + c3*t*t*t + c4*t*t*t*t

        def dP(t):
            return c1 + 2*c2*t + 3*c3*t*t + 4*c4*t*t*t

        # Start Newton iteration at t=0 (which is x0)
        t = 0.0
        best_x = x0
        best_f = abs(fx0)
        
        for _ in range(50):
            if time.time() > deadline:
                break
            pt = P(t)
            dpt = dP(t)
            if abs(dpt) < 1e-30:
                break
            t_new = t - pt / dpt
            # Check for divergence
            if abs(t_new) > 1e10:
                break
            if abs(t_new - t) < 1e-14 * max(1.0, abs(t)):
                t = t_new
                break
            t = t_new

        # Candidate root in x-space
        x_candidate = x0 + t * h
        # Verify with actual function call
        if time.time() < deadline:
            try:
                fx_candidate = f(x_candidate)
                if math.isfinite(fx_candidate) and abs(fx_candidate) <= 1e-8:
                    return x_candidate
                # If not good enough, we might try to refine with a few more Newton steps
                # using the actual function? No, that's expensive.
                # Instead, let's check if the model predicted a root but it's not exact.
                # We can try a few more points near x_candidate.
            except Exception:
                pass

    # If the Taylor model didn't yield a verified root, fall back to a simple
    # bracketing + bisection/secant approach.
    # Try to find a bracket around x0.
    h = 1e-4 * max(1.0, abs(x0))
    a, b = None, None
    fa, fb = None, None

    # We already have fx0 at x0.
    for _ in range(20):
        if time.time() > deadline:
            break
        x_left = x0 - h
        x_right = x0 + h
        try:
            f_left = f(x_left)
            if math.isfinite(f_left):
                if abs(f_left) <= 1e-8:
                    return x_left
                if fx0 * f_left < 0:
                    a, b = x_left, x0
                    fa, fb = f_left, fx0
                    break
        except Exception:
            pass

        try:
            f_right = f(x_right)
            if math.isfinite(f_right):
                if abs(f_right) <= 1e-8:
                    return x_right
                if fx0 * f_right < 0:
                    a, b = x0, x_right
                    fa, fb = fx0, f_right
                    break
        except Exception:
            pass

        h *= 10.0
        if h > 1e10:
            break

    if a is not None:
        # Use bisection with a cap on iterations
        for _ in range(100):
            if time.time() > deadline:
                break
            mid = (a + b) / 2.0
            try:
                f_mid = f(mid)
                if not math.isfinite(f_mid):
                    break
                if abs(f_mid) <= 1e-8:
                    return mid
                if fa * f_mid < 0:
                    b = mid
                    fb = f_mid
                else:
                    a = mid
                    fa = f_mid
            except Exception:
                break

        # Return the midpoint as the best guess
        return (a + b) / 2.0

    # If no bracket found, return x0
    return x0
