"""Root finder: Inverse quadratic interpolation with adaptive bracketing for smooth polynomials.

Structural change from secant-Newton hybrid to IQUICNE-style approach:
  1. Build a bracket by expanding from x0 (cheap, ~4-8 calls).
  2. Use inverse quadratic interpolation (IQI) within the bracket for fast convergence.
  3. Fall back to bisection when IQI step is unsafe.
  4. Track best x; return best if budget exhausted.
"""
import time
import math

_TOL = 1e-8
_CALL_CAP = 490
_DEADLINE_S = 140.0


def find_root(f, x0):
    x0 = float(x0)
    t_start = time.monotonic()
    calls = 0
    best_x = x0
    best_abs_f = float('inf')

    def _time_ok():
        return (time.monotonic() - t_start) < _DEADLINE_S

    def _fx(x):
        nonlocal calls, best_x, best_abs_f
        if calls >= _CALL_CAP:
            return float('nan')
        calls += 1
        try:
            v = float(f(x))
        except Exception:
            return float('nan')
        if v != v:  # NaN
            return v
        av = abs(v)
        if av < best_abs_f:
            best_abs_f = av
            best_x = x
        return v

    # Phase 0: Initial evaluation
    f0 = _fx(x0)
    if f0 != f0:
        return x0
    if abs(f0) <= _TOL:
        return x0

    # Phase 1: Build a bracket by expanding from x0
    # Try to find a sign change
    center = x0
    fc = f0
    scale = max(1.0, abs(center)) * 0.5

    a = b = None
    fa = fb = None
    found_bracket = False

    # First try small steps
    for m in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]:
        if not _time_ok() or calls >= _CALL_CAP - 2:
            break
        for sign in [1, -1]:
            x_try = center + sign * m * scale
            f_try = _fx(x_try)
            if f_try != f_try:
                continue
            if abs(f_try) <= _TOL:
                return x_try
            if fc * f_try < 0:
                if abs(fc) < abs(f_try):
                    a, fa = center, fc
                    b, fb = x_try, f_try
                else:
                    a, fa = x_try, f_try
                    b, fb = center, fc
                found_bracket = True
                break
        if found_bracket:
            break

    if not found_bracket:
        # No bracket found; try secant method as fallback
        return _secant_fallback(f, x0, f0, calls, best_x, best_abs_f, t_start)

    # Ensure |fa| <= |fb|
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    # Phase 2: IQUICNE-style iteration
    # We have bracket [a, b] with fa*fb < 0
    # Track three points for IQI: a (left), b (right), c (best interior)
    # Start with c = a (the point with smaller |f|)
    ca = a
    fa_c = fa
    cb = b
    fb_c = fb

    # We'll use three points: x1, x2, x3 with f1, f2, f3
    # x1 = a, x2 = b, x3 = a initially (degenerate)
    x1, f1 = a, fa
    x2, f2 = b, fb
    x3, f3 = a, fa  # duplicate for start

    for i in range(200):
        if not _time_ok() or calls >= _CALL_CAP - 1:
            break

        # Check if any point is already a root
        if abs(f1) <= _TOL:
            return x1
        if abs(f2) <= _TOL:
            return x2
        if abs(f3) <= _TOL:
            return x3

        # If x1 == x3, we only have two distinct points; use secant
        if x1 == x3 or x2 == x3:
            # Secant step from the two distinct points
            if x1 == x2:
                break
            # Use the two points with smaller |f|
            if abs(f1) < abs(f2):
                x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            else:
                x_new = x2 - f2 * (x1 - x2) / (f1 - f2)
        else:
            # Inverse quadratic interpolation
            # Solve for x such that f(x) = 0 using quadratic through (x1,f1), (x2,f2), (x3,f3)
            # Lagrange form: x_new = (x1*x2*f3*(f3-f2) + x2*x3*f1*(f1-f3) + x3*x1*f2*(f2-f1)) / 
            #                (f1*f2*(f1-f3) + f2*f3*(f2-f1) + f3*f1*(f3-f2))
            # But this can be unstable. Use the more stable formulation.
            
            # Standard IQI formula:
            # x_new = x1 + (x2-x1) * f1 * (f3*(x3-x1) - f2*(x2-x1)) / ... 
            # Let me use the standard inverse quadratic interpolation formula.
            
            # From Numerical Recipes:
            # p = x1 + (x2-x1)*f1*(f3*(x3-x1) - f2*(x2-x1)) / ...
            
            # Actually, let me use a simpler approach:
            # Fit quadratic y = A*x^2 + B*x + C through the three points
            # and find its root closest to the current best point.
            
            # Using Lagrange interpolation:
            # L1(x) = (x-x2)(x-x3) / ((x1-x2)(x1-x3))
            # L2(x) = (x-x1)(x-x3) / ((x2-x1)(x2-x3))
            # L3(x) = (x-x1)(x-x2) / ((x3-x1)(x3-x2))
            # f(x) = f1*L1(x) + f2*L2(x) + f3*L3(x)
            # Set f(x) = 0 and solve quadratic.
            
            den1 = (x1 - x2) * (x1 - x3)
            den2 = (x2 - x1) * (x2 - x3)
            den3 = (x3 - x1) * (x3 - x2)
            
            if den1 == 0 or den2 == 0 or den3 == 0:
                # Degenerate, use secant
                if x1 == x2:
                    break
                x_new = x1 - f1 * (x2 - x1) / (f2 - f1)
            else:
                # Coefficients of the quadratic
                # A = f1/den1 + f2/den2 + f3/den3
                # B = -f1*(x2+x3)/den1 - f2*(x1+x3)/den2 - f3*(x1+x2)/den3
                # C = f1*x2*x3/den1 + f2*x1*x3/den2 + f3*x1*x2/den3
                
                A = f1 / den1 + f2 / den2 + f3 / den3
                B = -f1 * (x2 + x3) / den1 - f2 * (x1 + x3) / den2 - f3 * (x1 + x2) / den3
                C = f1 * x2 * x3 / den1 + f2 * x1 * x3 / den2 + f3 * x1 * x2 / den3
                
                if A == 0:
                    # Linear
                    if B == 0:
                        break
                    x_new = -C / B
                else:
                    disc = B * B - 4 * A * C
                    if disc < 0:
                        # No real root; use bisection
                        x_new = 0.5 * (a + b)
                    else:
                        sq_disc = math.sqrt(disc)
                        r1 = (-B + sq_disc) / (2 * A)
                        r2 = (-B - sq_disc) / (2 * A)
                        # Pick the root closer to the point with smaller |f|
                        # and within the bracket
                        if abs(f1) < abs(f2):
                            ref = x1
                        else:
                            ref = x2
                        d1 = abs(r1 - ref)
                        d2 = abs(r2 - ref)
                        x_new = r1 if d1 <= d2 else r2
        
        # Ensure x_new is within the bracket
        if x_new <= a or x_new >= b:
            x_new = 0.5 * (a + b)
        
        # Additional safety: don't take steps too close to existing points
        min_dist = 1e-14 * max(1.0, abs(x_new))
        if abs(x_new - x1) < min_dist and abs(x_new - x2) < min_dist and abs(x_new - x3) < min_dist:
            x_new = 0.5 * (a + b)

        f_new = _fx(x_new)
        if f_new != f_new:
            break
        
        if abs(f_new) <= _TOL:
            return x_new

        # Update bracket
        if fa * f_new < 0:
            b = x_new
            fb = f_new
        elif fb * f_new < 0:
            a = x_new
            fa = f_new
        # Else x_new is outside bracket or same sign; ignore for bracketing

        # Update the three points for IQI
        # Keep the three points with smallest |f| values
        points = [(x1, f1), (x2, f2), (x3, f3), (x_new, f_new)]
        points.sort(key=lambda p: abs(p[1]))
        # Keep at least one from each end of bracket
        # Simple approach: keep the three smallest |f|, but ensure bracket endpoints are represented
        # For simplicity, just keep three smallest
        x1, f1 = points[0][0], points[0][1]
        x2, f2 = points[1][0], points[1][1]
        x3, f3 = points[2][0], points[2][1]

        # Check convergence
        if abs(b - a) < 1e-14:
            break

    # Return best
    if abs(fa) <= _TOL:
        return a
    if abs(fb) <= _TOL:
        return b
    return best_x


def _secant_fallback(f, x0, f0, calls, best_x, best_abs_f, t_start):
    """Fallback secant method when no bracket is found."""
    h = 1e-6 * max(1.0, abs(x0))
    x1 = x0
    f1 = f0
    x2 = x0 + h
    calls_local = calls
    best_x_local = best_x
    best_abs_f_local = best_abs_f

    def _fx_local(x):
        nonlocal calls_local, best_x_local, best_abs_f_local
        if calls_local >= _CALL_CAP:
            return float('nan')
        calls_local += 1
        try:
            v = float(f(x))
        except Exception:
            return float('nan')
        if v != v:
            return v
        av = abs(v)
        if av < best_abs_f_local:
            best_abs_f_local = av
            best_x_local = x
        return v

    f2 = _fx_local(x2)
    if f2 != f2:
        return best_x_local
    if abs(f2) <= _TOL:
        return x2

    for i in range(50):
        if (time.monotonic() - t_start) >= _DEADLINE_S or calls_local >= _CALL_CAP - 1:
            break
        if x2 == x1:
            break
        slope = (f2 - f1) / (x2 - x1)
        if slope == 0.0 or slope != slope:
            break
        dx = -f1 / slope
        x_new = x1 + dx
        max_step = 0.5 * max(1.0, abs(x1))
        if abs(dx) > max_step:
            dx = max_step * (1 if dx > 0 else -1)
            x_new = x1 + dx
        if x_new == x1:
            break
        f_new = _fx_local(x_new)
        if f_new != f_new:
            break
        if abs(f_new) <= _TOL:
            return x_new
        if abs(f_new) < abs(f1):
            x1, f1 = x_new, f_new
        else:
            x2, f2 = x_new, f_new

    return best_x_local
