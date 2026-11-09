"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Finds a root of f using a hybrid strategy:
    1. Try secant method (fast quadratic convergence near root).
    2. If secant fails or diverges, use a bracketing search to find sign change,
       then refine with Brent's method (inverse quadratic + bisection fallback).
    3. Strict deadline guard to prevent timeouts.
    """
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s kill

    x = float(x0)
    best_x = x
    best_abs = float('inf')

    def update_best(x_val, f_val):
        nonlocal best_x, best_abs
        af = abs(f_val)
        if af < best_abs:
            best_abs = af
            best_x = x_val

    def time_up():
        return (time.time() - start_time) > (deadline - start_time) - 0.3

    # Phase 1: Secant method
    f_x = f(x)
    update_best(x, f_x)
    if abs(f_x) <= 1e-8:
        return x

    # Initialize secant with a small perturbation
    h = 1e-8 * max(1.0, abs(x))
    x2 = x + h
    f_x2 = f(x2)
    update_best(x2, f_x2)
    if abs(f_x2) <= 1e-8:
        return x2

    x1 = x
    f_x1 = f_x
    x = x2
    f_x = f_x2

    # Secant method iterations
    for i in range(30):
        if time_up():
            break

        dx = x - x1
        if dx == 0 or not math.isfinite(dx):
            break
        d = (f_x - f_x1) / dx
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - f_x / d

        # Check if step is reasonable
        if not math.isfinite(x_new):
            break
        # Limit step size to avoid divergence
        max_step = 5.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            break

        x1 = x
        f_x1 = f_x
        x = x_new
        f_x = f(x)
        update_best(x, f_x)

        if abs(f_x) <= 1e-8:
            return x

        # Divergence check
        if abs(f_x) > abs(f_x1) * 3.0:
            break

    # Phase 2: Bracketing search if secant didn't converge
    a = best_x
    fa = f(a)
    if abs(fa) <= 1e-8:
        return a

    # Try expanding in both directions to find a sign change
    found_bracket = False
    step = 1.0
    for _ in range(50):
        if time_up():
            break
        for direction in [1, -1]:
            if time_up():
                break
            b = a + direction * step
            fb = f(b)
            update_best(b, fb)
            if abs(fb) <= 1e-8:
                return b
            if fa * fb < 0:
                found_bracket = True
                break
        if found_bracket:
            break
        # Update a to the point with smaller |f|
        if abs(fb) < abs(fa):
            a = b
            fa = fb
        step *= 2.0
        if step > 1e8:
            break

    if found_bracket:
        # Use Brent's method for guaranteed fast convergence
        # Inverse quadratic interpolation with bisection fallback
        for i in range(80):
            if time_up():
                break
            if fa * fb > 0:
                break
            
            # Inverse quadratic interpolation
            denom = (fa - fb)
            if abs(denom) > 1e-30:
                c = b - fb * (b - a) / (fb - fa)
            else:
                c = (a + b) / 2.0
            
            # Check if c is in bracket; if not, use bisection
            if c <= a or c >= b or not math.isfinite(c):
                c = (a + b) / 2.0
            
            fc = f(c)
            update_best(c, fc)
            if abs(fc) <= 1e-8:
                return c
            
            if fa * fc < 0:
                b = c
                fb = fc
            else:
                a = c
                fa = fc
            
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break

        # Return the best found
        if abs(fa) <= abs(fb):
            return a
        else:
            return b

    # If no bracket found, return best_x
    return best_x
# EVOLVE-BLOCK-END
