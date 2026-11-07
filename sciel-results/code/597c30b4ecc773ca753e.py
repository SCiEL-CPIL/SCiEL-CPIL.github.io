"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 140.0  # safety margin before 150s wall-clock
    x = float(x0)

    # Phase 1: Try Newton's method with a few iterations to get close
    # Use secant method to save function calls (1 eval per step after first)
    x0_val = f(x)
    if abs(x0_val) <= 1e-8:
        return x

    # Try a few secant steps to bracket or converge
    x_prev = x
    x_curr = x
    f_prev = x0_val

    # Establish a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x_curr = x + h
    f_curr = f(x_curr)

    # If we have a bracket, use safeguarded bisection/secant hybrid
    # Otherwise continue with secant
    max_iter = 40
    for i in range(max_iter):
        if time.time() > deadline:
            break
        if abs(f_curr) <= 1e-8:
            x = x_curr
            break
        if abs(f_prev) <= 1e-8:
            x = x_prev
            break

        # Check for bracket
        if f_prev * f_curr < 0:
            # We have a bracket [x_prev, x_curr]
            # Use inverse quadratic interpolation or bisection
            # Simple approach: secant step, check if in bracket
            denom = f_curr - f_prev
            if abs(denom) < 1e-30:
                x_new = 0.5 * (x_prev + x_curr)
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / denom

            # Safeguard: if x_new is outside bracket, bisect
            if x_new <= min(x_prev, x_curr) or x_new >= max(x_prev, x_curr):
                x_new = 0.5 * (x_prev + x_curr)

            # Update bracket
            f_new = f(x_new)
            if f_prev * f_new <= 0:
                x_curr = x_new
                f_curr = f_new
            else:
                x_prev = x_new
                f_prev = f_new
            continue
        else:
            # No bracket: use secant step
            denom = f_curr - f_prev
            if abs(denom) < 1e-30:
                # Derivative ~0, try a small perturbation
                h2 = 1e-6 * max(1.0, abs(x_curr))
                x_new = x_curr + h2
            else:
                x_new = x_curr - f_curr * (x_curr - x_prev) / denom

            f_new = f(x_new)
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_new
            f_curr = f_new
            x = x_curr

    # Phase 2: Polish pass with tighter tolerance
    # Use Newton's method with finite differences for final refinement
    best_x = x
    best_fx = f(x)
    for _ in range(30):
        if time.time() > deadline:
            break
        if abs(best_fx) <= 1e-12:
            break
        h = 1e-8 * max(1.0, abs(best_x))
        d = (f(best_x + h) - best_fx) / h
        if abs(d) < 1e-30:
            break
        step = best_fx / d
        # Limit step size to avoid overshooting
        max_step = 0.5 * max(1.0, abs(best_x))
        if abs(step) > max_step:
            step = max_step * (1.0 if step > 0 else -1.0)
        new_x = best_x - step
        new_fx = f(new_x)
        if abs(new_fx) < abs(best_fx):
            best_x = new_x
            best_fx = new_fx
        else:
            # Try smaller step
            best_x = best_x - step * 0.5
            best_fx = f(best_x)

    return best_x
# EVOLVE-BLOCK-END
