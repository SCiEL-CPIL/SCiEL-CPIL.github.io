import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Strategy:
      1. Check x0 immediately.
      2. Try secant/Newton to converge fast if close.
      3. If secant stalls or diverges, do a bracketing search
         (Illinois / bisection hybrid) to guarantee convergence.
      4. Final polish with tight Newton if needed.
    """
    start_time = time.time()
    deadline = start_time + 12.0
    tol = 1e-8
    tol_inner = 1e-14
    max_calls = 480  # safety cap

    call_count = 0

    def call(x):
        nonlocal call_count
        call_count += 1
        return f(x)

    def time_ok():
        return time.time() < deadline and call_count < max_calls

    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 1.0

    f0 = call(x0)
    if math.isfinite(f0) and abs(f0) <= tol:
        return x0

    # Handle non-finite f0
    if not math.isfinite(f0):
        base = 1e-4 * max(1.0, abs(x0))
        found = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128]:
            if not time_ok():
                break
            x_try = x0 + s * base
            f_try = call(x_try)
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                found = True
                if abs(f0) <= tol:
                    return x0
                break
        if not found or not math.isfinite(f0):
            return x0

    # --- Phase 1: Secant / Newton-style iteration ---
    # We need two points. Use x0 and a small perturbation.
    h = 1e-6 * max(1.0, abs(x0))
    xa = x0
    fa = f0
    xb = x0 + h
    fb = call(xb)

    if not math.isfinite(fb):
        xb = x0 - h
        fb = call(xb)

    if not math.isfinite(fb):
        # Try a few more steps
        for s in [2, -2, 4, -4, 8, -8]:
            if not time_ok():
                break
            xb = x0 + s * h * 10
            fb = call(xb)
            if math.isfinite(fb):
                break
        if not math.isfinite(fb):
            return xa

    best_x = xa if abs(fa) < abs(fb) else xb
    best_f = min(abs(fa), abs(fb))

    if abs(fa) <= tol:
        return xa
    if abs(fb) <= tol:
        return xb

    # Secant iterations
    secant_ok = True
    for it in range(40):
        if not time_ok():
            break
        denom = fb - fa
        if abs(denom) < 1e-15:
            break
        xn = xb - fb * (xb - xa) / denom
        if not math.isfinite(xn):
            secant_ok = False
            break
        fn = call(xn)
        if not math.isfinite(fn):
            secant_ok = False
            break
        if abs(fn) < best_f:
            best_x = xn
            best_f = abs(fn)
        if abs(fn) <= tol:
            return xn
        # Keep the two points; update
        xa, fa = xb, fb
        xb, fb = xn, fn
        if abs(xn - xb) < 1e-15 and abs(xn - xa) < 1e-15:
            break

    if secant_ok and abs(fb) <= tol:
        return xb
    if secant_ok and abs(fa) <= tol:
        return xa

    # --- Phase 2: Bracketing search (Illinois method) ---
    # Need to find a sign change. Start from the current best region.
    # We'll scan outward from the best point found so far.

    # First, try to bracket around best_x
    a = best_x
    fa_best = call(a) if abs(fa_best := f(a)) > tol else None
    # Actually, let's just re-evaluate to be safe, but we should cache.
    # Let's keep a small cache.

    # Simple approach: expand interval from best_x
    # We already have some points. Let's find two with opposite signs.
    # Use the points we have: (xa, fa), (xb, fb), (best_x, best_f)
    # Collect all known finite points
    points = [(xa, fa), (xb, fb)]
    # Add best_x if distinct
    if abs(best_x - xa) > 1e-15 and abs(best_x - xb) > 1e-15:
        points.append((best_x, best_f))

    # Sort by x
    points.sort(key=lambda p: p[0])

    # Check for existing sign change
    bracket = None
    for i in range(len(points) - 1):
        x1, f1 = points[i]
        x2, f2 = points[i + 1]
        if f1 * f2 < 0:
            bracket = (x1, f1, x2, f2)
            break

    if bracket is None:
        # Expand outward from the best point
        center = best_x
        f_center = best_f
        step = h
        for attempt in range(60):
            if not time_ok():
                break
            # Try expanding in both directions
            for direction in [1, -1]:
                x_new = center + direction * step
                f_new = call(x_new)
                if not math.isfinite(f_new):
                    continue
                # Check sign change with center
                if f_center * f_new < 0:
                    if direction > 0:
                        bracket = (center, f_center, x_new, f_new)
                    else:
                        bracket = (x_new, f_new, center, f_center)
                    break
                # Also check with previous points
                for px, pf in points:
                    if f_new * pf < 0:
                        if px < x_new:
                            bracket = (px, pf, x_new, f_new)
                        else:
                            bracket = (x_new, f_new, px, pf)
                        break
                if bracket:
                    break
            if bracket:
                break
            step *= 2

    if bracket is None:
        # Couldn't bracket. Return best found.
        return best_x

    a, fa, b, fb = bracket
    # Ensure fa < 0, fb > 0 for standard bisection/Illinois
    if fa > 0:
        a, fa, b, fb = b, fb, a, fa

    # Illinois method
    for it in range(60):
        if not time_ok():
            break
        # Illinois: weight the endpoints based on |f|
        if abs(fa) < abs(fb):
            # fa is smaller, move b
            if fa < 0:
                # a is negative, b is positive
                # New b: interpolate
                b_new = b - fb * (b - a) / (fb - fa)
                if b_new <= a or b_new >= b:
                    b_new = (a + b) / 2.0
                fb_new = call(b_new)
                if not math.isfinite(fb_new):
                    b_new = (a + b) / 2.0
                    fb_new = call(b_new)
                if fb_new * fa < 0:
                    # Root between a and b_new
                    if abs(fb_new) < abs(fb):
                        b, fb = b_new, fb_new
                    else:
                        b, fb = b_new, fb_new
                    # Illinois: shrink the larger interval
                    if abs(fa) < abs(fb):
                        b = (a + b) / 2.0
                        fb = call(b)
                else:
                    # fa and fb_new same sign, root between b_new and b
                    a, fa = b_new, fb_new
                    if abs(fa) < abs(fb):
                        a = (a + b) / 2.0
                        fa = call(a)
                if abs(fa) <= tol:
                    return a
                if abs(fb) <= tol:
                    return b
            else:
                # This case shouldn't happen since we ensured fa<0
                b_new = (a + b) / 2.0
                fb_new = call(b_new)
                if fb_new * fa < 0:
                    b, fb = b_new, fb_new
                else:
                    a, fa = b_new, fb_new
        else:
            # fb is smaller, move a
            if fb > 0:
                a_new = a - fa * (b - a) / (fb - fa)
                if a_new >= b or a_new <= a:
                    a_new = (a + b) / 2.0
                fa_new = call(a_new)
                if not math.isfinite(fa_new):
                    a_new = (a + b) / 2.0
                    fa_new = call(a_new)
                if fa_new * fb < 0:
                    a, fa = a_new, fa_new
                    if abs(fa) < abs(fb):
                        a = (a + b) / 2.0
                        fa = call(a)
                else:
                    b, fb = a_new, fa_new
                    if abs(fb) < abs(fa):
                        b = (a + b) / 2.0
                        fb = call(b)
                if abs(fa) <= tol:
                    return a
                if abs(fb) <= tol:
                    return b
            else:
                a_new = (a + b) / 2.0
                fa_new = call(a_new)
                if fa_new * fb < 0:
                    a, fa = a_new, fa_new
                else:
                    b, fb = a_new, fa_new

        # Convergence check
        if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
            mid = (a + b) / 2.0
            return mid

    # Return the better endpoint
    if abs(fa) <= abs(fb):
        return a
    return b
