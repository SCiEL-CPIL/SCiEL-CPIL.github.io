"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, using a robust portfolio approach.
    Uses deadline guard to stay within wall-clock budget.
    """
    deadline = time.time() + 145.0
    tol = 1e-9  # target |f(x)| <= 1e-8, use tighter internal tol
    
    x = float(x0)
    
    # Best solution found so far
    best_x = x
    best_f = f(x)
    best_abs = abs(best_f)
    
    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(x_val) or f_val is None:
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f
    
    def check_deadline():
        return time.time() >= deadline
    
    # Phase 1: Try Newton's method with finite-difference derivative
    x_newton = float(x0)
    for _ in range(50):
        if check_deadline():
            break
        try:
            fx = f(x_newton)
            if fx is None:
                break
            update_best(x_newton, fx)
            if abs(fx) <= tol:
                return x_newton
            h = 1e-6 * max(1.0, abs(x_newton))
            d = (f(x_newton + h) - fx) / h
            if d == 0 or not math.isfinite(d):
                break
            x_newton = x_newton - fx / d
            if not math.isfinite(x_newton):
                break
        except:
            break
    
    # Phase 2: Try bisection method with bracketing
    # First, try to find a bracket around x0
    a = float(x0)
    fa = f(a)
    if fa is None or not math.isfinite(fa):
        fa = 0.0
    
    # Try expanding bracket
    bracket_found = False
    step = max(1.0, abs(x0))
    for _ in range(20):
        if check_deadline():
            break
        b = a + step
        fb = f(b)
        if fb is None or not math.isfinite(fb):
            step *= 2
            continue
        if fa * fb <= 0:
            bracket_found = True
            break
        # If fa is zero, we already have a root
        if abs(fa) <= tol:
            return a
        step *= 2
        if step > 1e6:
            break
    
    if bracket_found:
        # Bisection
        for _ in range(100):
            if check_deadline():
                break
            mid = (a + b) / 2.0
            fm = f(mid)
            if fm is None or not math.isfinite(fm):
                break
            update_best(mid, fm)
            if abs(fm) <= tol:
                return mid
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
                break
    
    # Phase 3: Try secant method
    x0_sec = float(x0)
    x1_sec = x0_sec + max(1e-3, 1e-3 * abs(x0_sec))
    fx0 = f(x0_sec)
    fx1 = f(x1_sec)
    if fx0 is not None and fx1 is not None and math.isfinite(fx0) and math.isfinite(fx1):
        for _ in range(50):
            if check_deadline():
                break
            if abs(fx0) <= tol:
                return x0_sec
            if abs(fx1) <= tol:
                return x1_sec
            if fx0 == fx1:
                break
            x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / (fx1 - fx0)
            if not math.isfinite(x2_sec):
                break
            fx2 = f(x2_sec)
            if fx2 is None or not math.isfinite(fx2):
                break
            update_best(x2_sec, fx2)
            x0_sec = x1_sec
            fx0 = fx1
            x1_sec = x2_sec
            fx1 = fx2
    
    # Phase 4: Try simple iteration / fixed point if other methods failed
    # Use a small step toward where f is decreasing
    x_iter = float(x0)
    for _ in range(100):
        if check_deadline():
            break
        fx = f(x_iter)
        if fx is None or not math.isfinite(fx):
            break
        update_best(x_iter, fx)
        if abs(fx) <= tol:
            return x_iter
        # Estimate derivative
        h = 1e-5 * max(1.0, abs(x_iter))
        d = (f(x_iter + h) - fx) / h
        if d == 0 or not math.isfinite(d):
            # Try a small step in the direction of -sign(fx)
            step = 1e-3 * max(1.0, abs(x_iter))
            x_iter = x_iter - math.copysign(step, fx)
        else:
            # Damped Newton step
            step = fx / d
            if abs(step) > max(1.0, abs(x_iter)):
                step = math.copysign(max(1.0, abs(x_iter)), step)
            x_iter = x_iter - step
        if not math.isfinite(x_iter):
            break
    
    # Return the best solution found
    if best_abs <= 1e-8:
        return best_x
    return best_x
# EVOLVE-BLOCK-END
