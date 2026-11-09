"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    def time_up():
        return time.time() >= deadline
    
    x0 = float(x0)
    
    # Track best solution found
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        ax = abs(fx)
        if ax < best_abs:
            best_x = x
            best_f = fx
            best_abs = ax
    
    # If already good enough
    if best_abs <= 1e-8:
        return best_x
    
    # Try to bracket a root using expanding interval
    h = 1e-3 * max(1.0, abs(x0))
    a = x0 - h
    b = x0 + h
    fa = f(a)
    fb = f(b)
    update_best(a, fa)
    update_best(b, fb)
    
    bracket_found = False
    if fa == 0:
        return a
    if fb == 0:
        return b
    if fa * fb < 0:
        bracket_found = True
    else:
        # Expand interval to try to find a sign change
        for i in range(50):
            if time_up():
                break
            h *= 2
            a = x0 - h
            b = x0 + h
            fa = f(a)
            fb = f(b)
            update_best(a, fa)
            update_best(b, fb)
            if fa == 0:
                return a
            if fb == 0:
                return b
            if fa * fb < 0:
                bracket_found = True
                break
    
    # Also try a one-sided expansion if no bracket found
    if not bracket_found:
        # Try expanding in both directions separately
        for direction in [1, -1]:
            if time_up():
                break
            h = 1e-3 * max(1.0, abs(x0))
            for i in range(50):
                if time_up():
                    break
                x_new = x0 + direction * h
                fx_new = f(x_new)
                update_best(x_new, fx_new)
                if abs(fx_new) <= 1e-8:
                    return x_new
                if fx_new == 0:
                    return x_new
                if best_f * fx_new < 0:
                    # We have a bracket between x0 and x_new
                    if direction > 0:
                        a, b, fa, fb = x0, x_new, best_f, fx_new
                    else:
                        a, b, fa, fb = x_new, x0, fx_new, best_f
                    bracket_found = True
                    break
                h *= 2
            if bracket_found:
                break
    
    if not bracket_found:
        # No bracket found; try Newton's method as fallback
        x = x0
        for _ in range(100):
            if time_up():
                break
            fx = f(x)
            update_best(x, fx)
            if abs(fx) <= 1e-8:
                return x
            if abs(fx) <= 1e-14:
                break
            h = 1e-7 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0 or abs(d) < 1e-15:
                break
            x_new = x - fx / d
            # Damping if step is too large
            if abs(x_new - x) > 10 * h:
                x_new = x + math.copysign(10 * h, x_new - x)
            x = x_new
        return best_x
    
    # Bisection + secant hybrid on the bracket [a, b]
    # Use secant when safe, fall back to bisection
    x1 = a
    x2 = b
    f1 = fa
    f2 = fb
    
    for _ in range(200):
        if time_up():
            break
        
        # Check if either endpoint is good enough
        if abs(f1) <= 1e-8:
            return x1
        if abs(f2) <= 1e-8:
            return x2
        
        # Secant step
        if f1 != f2:
            x_sec = x1 - f1 * (x2 - x1) / (f2 - f1)
        else:
            x_sec = (x1 + x2) / 2
        
        # Check if secant point is within bracket
        if x1 < x_sec < x2:
            f_sec = f(x_sec)
            update_best(x_sec, f_sec)
            if abs(f_sec) <= 1e-8:
                return x_sec
            if f_sec == 0:
                return x_sec
            
            # Update bracket
            if f1 * f_sec < 0:
                x2 = x_sec
                f2 = f_sec
            else:
                x1 = x_sec
                f1 = f_sec
        else:
            # Bisection
            x_mid = (x1 + x2) / 2
            f_mid = f(x_mid)
            update_best(x_mid, f_mid)
            if abs(f_mid) <= 1e-8:
                return x_mid
            if f_mid == 0:
                return x_mid
            
            if f1 * f_mid < 0:
                x2 = x_mid
                f2 = f_mid
            else:
                x1 = x_mid
                f1 = f_mid
        
        # Convergence check
        if abs(x2 - x1) < 1e-15:
            break
    
    return best_x
# EVOLVE-BLOCK-END
