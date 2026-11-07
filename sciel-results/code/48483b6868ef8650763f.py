"""Rewritten root finder with improved strategy."""
import time
import math

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin under 120s

    x = float(x0)
    call_count = 0

    # Get first evaluation
    try:
        fx = f(x)
        call_count += 1
    except Exception:
        return x

    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx

    # Try to find a good second point for secant
    step_factors = [1e-4, 1e-3, 1e-2, 0.1, 1.0, -1e-4, -1e-3, -1e-2, -0.1, -1.0]
    x2 = None
    fx2 = None

    for sf in step_factors:
        if time.time() > deadline:
            return best_x
        h = sf * max(1.0, abs(x))
        x_try = x + h
        try:
            fx_try = f(x_try)
            call_count += 1
        except Exception:
            continue
        if not math.isfinite(fx_try):
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        if abs(fx_try) < abs(best_fx):
            best_x = x_try
            best_fx = fx_try
        # Accept if we have a reasonable slope
        if abs(fx_try - fx) > 1e-15 * max(1.0, abs(fx), abs(fx_try)):
            x2 = x_try
            fx2 = fx_try
            break

    if x2 is None or fx2 is None:
        # Couldn't find a good second point, try bisection-like approach
        # Try a few more points with larger steps
        for scale in [10.0, 100.0, 1000.0]:
            if time.time() > deadline:
                return best_x
            h = scale * max(1.0, abs(x))
            for sign in [1.0, -1.0]:
                x_try = x + sign * h
                try:
                    fx_try = f(x_try)
                    call_count += 1
                except Exception:
                    continue
                if not math.isfinite(fx_try):
                    continue
                if abs(fx_try) <= 1e-8:
                    return x_try
                if abs(fx_try) < abs(best_fx):
                    best_x = x_try
                    best_fx = fx_try
                if abs(fx_try - fx) > 1e-15 * max(1.0, abs(fx), abs(fx_try)):
                    x2 = x_try
                    fx2 = fx_try
                    break
            if x2 is not None:
                break

    if x2 is None or fx2 is None:
        return best_x

    # Secant method with stall detection
    prev_best = abs(best_fx)
    stall_count = 0
    max_iter = 498 - call_count
    if max_iter < 10:
        max_iter = 10

    for i in range(max_iter):
        if time.time() > deadline:
            return best_x

        denom = fx2 - fx
        if abs(denom) < 1e-30:
            # Nearly parallel; perturb
            h2 = 1e-2 * max(1.0, abs(x2))
            x_new = x2 + h2
            try:
                fx_new = f(x_new)
                call_count += 1
            except Exception:
                return best_x
            if not math.isfinite(fx_new):
                return best_x
            if abs(fx_new) < abs(best_fx):
                best_x = x_new
                best_fx = fx_new
            if abs(fx_new) <= 1e-8:
                return x_new
            x, fx = x2, fx2
            x2, fx2 = x_new, fx_new
            stall_count = 0
            continue

        x_new = x2 - fx2 * (x2 - x) / denom
        if not math.isfinite(x_new):
            return best_x

        try:
            fx_new = f(x_new)
            call_count += 1
        except Exception:
            return best_x

        if not math.isfinite(fx_new):
            return best_x

        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
            stall_count = 0
        else:
            stall_count += 1

        if abs(fx_new) <= 1e-8:
            return x_new

        # If we've stalled (no improvement in last 5 steps), try a larger perturbation
        if stall_count >= 5:
            h3 = 0.1 * max(1.0, abs(best_x)) * (1.0 if (i % 2 == 0) else -1.0)
            x_pert = best_x + h3
            try:
                fx_pert = f(x_pert)
                call_count += 1
            except Exception:
                return best_x
            if math.isfinite(fx_pert):
                if abs(fx_pert) < abs(best_fx):
                    best_x = x_pert
                    best_fx = fx_pert
                if abs(fx_pert) <= 1e-8:
                    return x_pert
                x, fx = x2, fx2
                x2, fx2 = x_pert, fx_pert
            stall_count = 0
            continue

        # Shift: x = x2, x2 = x_new
        x, fx = x2, fx2
        x2, fx2 = x_new, fx_new

    return best_x
