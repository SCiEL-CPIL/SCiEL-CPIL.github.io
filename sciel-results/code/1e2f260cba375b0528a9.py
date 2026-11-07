"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    x = float(x0)
    
    # 1. Evaluate initial point
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # If non-finite, try to find a nearby finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    # Best point tracking
    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Initialize Secant Method
    # Choose a step size that balances local accuracy and avoiding numerical noise
    h = max(1e-5, abs(x) * 1e-5)
    
    x_prev = x
    f_prev = fx
    
    x_curr = x + h
    try:
        f_curr = f(x_curr)
    except Exception:
        x_curr = x - h
        try:
            f_curr = f(x_curr)
        except Exception:
            return best_x
        
    if not math.isfinite(f_curr):
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]:
            for sign in [1, -1]:
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x_curr = x_test
                    f_curr = fx_test
                    found = True
                    break
            if found:
                break
        if not found:
            return best_x

    if abs(f_curr) <= 1e-8:
        return x_curr
    if abs(f_curr) < best_abs_fx:
        best_x = x_curr
        best_fx = f_curr
        best_abs_fx = abs(f_curr)

    # Secant Iteration with improved safeguards
    max_iter = 40
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check for zero slope
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            # Perturb to escape
            x_curr = x_curr + 1e-4
            try:
                f_curr = f(x_curr)
            except Exception:
                break
            if not math.isfinite(f_curr):
                break
            if abs(f_curr) < best_abs_fx:
                best_x = x_curr
                best_fx = f_curr
                best_abs_fx = abs(f_curr)
            if abs(f_curr) <= 1e-8:
                return x_curr
            continue

        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        # Safeguard: limit step size to avoid jumping too far
        max_step = max(1.0, abs(x_curr) * 0.3)
        if abs(x_new - x_curr) > max_step:
            x_new = x_curr + math.copysign(max_step, x_new - x_curr)
            
        # Evaluate new point
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            # Bisection fallback
            x_new = 0.5 * (x_curr + x_prev)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        # Update best
        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)
            
        if abs(f_new) <= 1e-8:
            return x_new

        # Stagnation check
        if abs(x_new - x_curr) < 1e-13 * max(1.0, abs(x_curr)):
             # Small perturbation
             x_new = x_curr + 1e-5
             try:
                 f_new = f(x_new)
             except Exception:
                 break
             if not math.isfinite(f_new):
                 break
             if abs(f_new) < best_abs_fx:
                 best_x = x_new
                 best_fx = f_new
                 best_abs_fx = abs(f_new)
             if abs(f_new) <= 1e-8:
                 return x_new

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_new, f_new

    # Final polish: use Newton-like iteration if we have a good estimate
    # Use central difference for derivative estimate
    if best_abs_fx > 1e-8 and best_abs_fx < 1e-4:
        # Try a few Newton steps with estimated derivative
        x_polish = best_x
        f_polish = best_fx
        for _ in range(5):
            if time.time() > deadline:
                break
            # Central difference for derivative
            h_deriv = max(1e-8, abs(x_polish) * 1e-8)
            try:
                f_plus = f(x_polish + h_deriv)
                f_minus = f(x_polish - h_deriv)
            except Exception:
                break
            if not (math.isfinite(f_plus) and math.isfinite(f_minus)):
                break
            df = (f_plus - f_minus) / (2 * h_deriv)
            if abs(df) < 1e-15:
                break
            x_newton = x_polish - f_polish / df
            try:
                f_newton = f(x_newton)
            except Exception:
                break
            if not math.isfinite(f_newton):
                break
            if abs(f_newton) < abs(f_polish):
                x_polish = x_newton
                f_polish = f_newton
                if abs(f_newton) < best_abs_fx:
                    best_x = x_polish
                    best_fx = f_newton
                    best_abs_fx = abs(f_newton)
                if abs(f_newton) <= 1e-8:
                    return x_newton
            else:
                break

    return best_x
# EVOLVE-BLOCK-END
