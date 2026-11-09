import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Structural change: Lead with secant/Newton (fast for close roots),
    fall back to bracketing with inverse-quadratic/bisection hybrid.
    """
    start_time = time.time()
    deadline = start_time + 10.0
    tol = 1e-12
    tol_final = 1e-8
    max_iter = 100

    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0
    if not math.isfinite(f0):
        # f0 is non-finite; need to find a finite point first
        base_step = 1e-4 * max(1.0, abs(x0))
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64]:
            x_try = x0 + s * base_step
            f_try = f(x_try)
            if math.isfinite(f_try) and abs(f_try) <= tol:
                return x_try
            if math.isfinite(f_try):
                x0, f0 = x_try, f_try
                break
        if not math.isfinite(f0):
            return x0

    # Phase 1: Secant/Newton hybrid — try to converge quickly
    # Use secant with a small perturbation
    base_step = 1e-6 * max(1.0, abs(x0))
    x_prev = x0
    f_prev = f0
    x_curr = x0 + base_step
    f_curr = f(x_curr)
    if abs(f_curr) <= tol:
        return x_curr
    if not math.isfinite(f_curr):
        # Try other perturbation directions
        x_curr = x0 - base_step
        f_curr = f(x_curr)
        if abs(f_curr) <= tol:
            return x_curr
        if not math.isfinite(f_curr):
            x_curr = x0 + base_step * 10
            f_curr = f(x_curr)
            if abs(f_curr) <= tol:
                return x_curr
            if not math.isfinite(f_curr):
                # Fall through to bracketing
                pass

    best_x = x0
    best_f = abs(f0)
    if abs(f_curr) < best_f:
        best_x, best_f = x_curr, abs(f_curr)

    # Secant iterations
    for _ in range(20):
        if time.time() > deadline:
            break
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            break
        x_next = x_curr - f_curr * (x_curr - x_prev) / denom
        if not math.isfinite(x_next):
            break
        f_next = f(x_next)
        if not math.isfinite(f_next):
            break
        if abs(f_next) < best_f:
            best_x, best_f = x_next, abs(f_next)
        if abs(f_next) <= tol_final:
            return x_next
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    if abs(f_curr) <= tol_final:
        return x_curr
    if abs(f_prev) <= tol_final:
        return x_prev

    # Phase 2: Bracketing with hybrid method (bisection + inverse quadratic)
    # We have two points (x_prev, f_prev) and (x_curr, f_curr)
    # Try to find a sign change
    a, fa = x_prev, f_prev
    b, fb = x_curr, f_curr

    if fa * fb > 0:
        # No sign change yet; expand to find one
        # Use adaptive expansion from the point with smaller |f|
        if abs(fa) < abs(fb):
            a, b = a, b
        else:
            a, b = b, a
        fa, fb = f(a), f(b)

        base_step2 = 1e-4 * max(1.0, abs(a))
        bracket_found = False
        for s in [1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64, 128, -128, 256, -256, 512, -512]:
            if time.time() > deadline:
                break
            x_try = a + s * base_step2
            f_try = f(x_try)
            if not math.isfinite(f_try):
                continue
            if abs(f_try) <= tol:
                return x_try
            if fa * f_try < 0:
                if abs(fa) < abs(f_try):
                    a, b, fa, fb = a, x_try, fa, f_try
                else:
                    a, b, fa, fb = x_try, a, f_try, fa
                bracket_found = True
                break

        if not bracket_found:
            # Return best found
            return best_x

    # Hybrid: Inverse Quadratic Interpolation with Bisection fallback
    # Track three points for IQI
    # Start with a, b and a midpoint
    m = (a + b) / 2.0
    fm = f(m)
    if not math.isfinite(fm):
        # Use bisection
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            if abs(fa) <= tol_final:
                return a
            if abs(fb) <= tol_final:
                return b
            if abs(b - a) < 1e-15:
                return (a + b) / 2.0
            mid = (a + b) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if fa * fmid < 0:
                b, fb = mid, fmid
            else:
                a, fa = mid, fmid
        return a if abs(fa) < abs(fb) else b

    if abs(fm) <= tol_final:
        return m

    # Now we have three points: (a,fa), (m,fm), (b,fb)
    # Use inverse quadratic interpolation when possible, else bisection
    x1, f1 = a, fa
    x2, f2 = m, fm
    x3, f3 = b, fb

    for _ in range(max_iter):
        if time.time() > deadline:
            break
        if abs(f1) <= tol_final:
            return x1
        if abs(f2) <= tol_final:
            return x2
        if abs(f3) <= tol_final:
            return x3
        if abs(x3 - x1) < 1e-15:
            return (x1 + x3) / 2.0

        # Determine which subinterval contains the root
        if f1 * f2 < 0:
            # Root in [x1, x2]
            x_lo, f_lo, x_hi, f_hi = x1, f1, x2, f2
            x_mid, f_mid = x3, f3
        elif f2 * f3 < 0:
            # Root in [x2, x3]
            x_lo, f_lo, x_hi, f_hi = x2, f2, x3, f3
            x_mid, f_mid = x1, f1
        else:
            # Shouldn't happen if bracket is maintained; use bisection
            mid = (x1 + x3) / 2.0
            fmid = f(mid)
            if not math.isfinite(fmid):
                break
            if abs(fmid) <= tol_final:
                return mid
            if f1 * fmid < 0:
                x3, f3 = mid, fmid
            else:
                x1, f1 = mid, fmid
            continue

        # Try inverse quadratic interpolation on the three points
        # IQI: fit quadratic to (f1,x1), (f2,x2), (f3,x3) and find where it crosses 0
        # Lagrange interpolation for x as function of f
        # x(f) = x1 * (f-f2)(f-f3)/((f1-f2)(f1-f3)) + x2 * (f-f1)(f-f3)/((f2-f1)(f2-f3)) + x3 * (f-f1)(f-f2)/((f3-f1)(f3-f2))
        # Set f=0:
        denom1 = (f_lo - f_mid) * (f_lo - f_hi)
        denom2 = (f_mid - f_lo) * (f_mid - f_hi)
        denom3 = (f_hi - f_lo) * (f_hi - f_mid)

        x_iqi = None
        if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
            x_iqi = (x_lo * f_mid * f_hi / denom1 +
                     x_mid * f_lo * f_hi / denom2 +
                     x_hi * f_lo * f_mid / denom3)

        # Check if IQI point is within the bracket
        if (x_iqi is not None and math.isfinite(x_iqi)
                and x_lo < x_iqi < x_hi):
            x_new = x_iqi
        else:
            # Bisection
            x_new = (x_lo + x_hi) / 2.0

        f_new = f(x_new)
        if not math.isfinite(f_new):
            # Fallback to bisection
            x_new = (x_lo + x_hi) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break

        if abs(f_new) <= tol_final:
            return x_new

        # Update the three points properly
        if f_lo * f_new < 0:
            # Root in [x_lo, x_new]
            x_hi, f_hi = x_new, f_new
        else:
            # Root in [x_new, x_hi]
            x_lo, f_lo = x_new, f_new

        # x_mid remains the non-bracketing point (the one not in [x_lo, x_hi])
        # Reassign: x1=x_lo, x2=x_mid, x3=x_hi
        x1, f1 = x_lo, f_lo
        x3, f3 = x_hi, f_hi
        x2, f2 = x_mid, f_mid

    # Final polish: if we have a bracket and haven't hit tol_final, do a few more bisections
    # to try to squeeze below 1e-8
    if time.time() < deadline:
        # Find current best bracket
        best_a, best_b = None, None
        best_fa, best_fb = None, None
        pts = [(x1, f1), (x2, f2), (x3, f3)]
        for i in range(3):
            for j in range(i+1, 3):
                if pts[i][1] * pts[j][1] < 0:
                    if best_a is None or abs(pts[i][1]) + abs(pts[j][1]) < abs(best_fa) + abs(best_fb):
                        best_a, best_fa = pts[i]
                        best_b, best_fb = pts[j]
        if best_a is not None and best_b is not None:
            a2, fa2 = best_a
            b2, fb2 = best_b
            for _ in range(20):
                if time.time() > deadline:
                    break
                if abs(fa2) <= tol_final:
                    return a2
                if abs(fb2) <= tol_final:
                    return b2
                if abs(b2 - a2) < 1e-15:
                    return (a2 + b2) / 2.0
                mid2 = (a2 + b2) / 2.0
                fm2 = f(mid2)
                if not math.isfinite(fm2):
                    break
                if abs(fm2) <= tol_final:
                    return mid2
                if fa2 * fm2 < 0:
                    b2, fb2 = mid2, fm2
                else:
                    a2, fa2 = mid2, fm2
            # Return whichever endpoint is closer to zero
            return a2 if abs(fa2) < abs(fb2) else b2

    # Return best
    candidates = [(x1, f1), (x2, f2), (x3, f3)]
    best = min(candidates, key=lambda t: abs(t[1]))
    return best[0]
