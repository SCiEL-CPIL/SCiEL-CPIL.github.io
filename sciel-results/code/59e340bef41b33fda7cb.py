"""Root finder for smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

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

    # Strategy: Use a robust combination of Newton and Secant methods with a line search
    # on |f|^2. For polynomials, Newton converges quadratically if close enough.
    # We'll maintain a history of points for Secant fallback.

    # History for Secant: list of (x, fx)
    history = [(x, fx)]

    # Parameters
    max_iter = 50
    h_fd = 1e-5  # Finite difference step

    for i in range(max_iter):
        if time.time() > deadline:
            break

        # 1. Try Newton step
        # Estimate derivative using central difference if possible (2 calls), else forward (1 call)
        # Since calls are counted, let's be conservative.
        # Actually, let's try to reuse history if possible for Secant first, or just Newton.
        # Newton is faster if derivative is accurate.

        # Let's try a hybrid:
        # If we have at least 2 points in history, try Secant.
        # Else, try Newton with FD derivative.

        x_new = None

        if len(history) >= 2:
            # Secant method
            x1, f1 = history[-1]
            x0_h, f0_h = history[-2]
            if abs(f1 - f0_h) > 1e-14:
                x_new = x1 - f1 * (x1 - x0_h) / (f1 - f0_h)

        if x_new is None or not math.isfinite(x_new):
            # Newton with FD derivative
            # Use forward difference to save a call? Central is better but 2 calls.
            # Let's use central difference if we have time budget, otherwise forward.
            # Given we want to minimize calls, forward difference (1 call) is safer.
            # But central difference gives better derivative.
            # Let's try: if |f| is large, use central (2 calls) for better direction.
            # If |f| is small, use forward (1 call) or secant.

            # Let's stick to forward difference for simplicity and call count, unless we are stuck.
            x_h = x + h_fd
            try:
                f_h = f(x_h)
            except Exception:
                # Try smaller h
                x_h = x + 1e-8
                try:
                    f_h = f(x_h)
                except Exception:
                    break
                if not math.isfinite(f_h):
                    break

            if math.isfinite(f_h):
                f_prime = (f_h - fx) / h_fd
                if math.isfinite(f_prime) and abs(f_prime) > 1e-14:
                    x_new = x - fx / f_prime

        if x_new is None or not math.isfinite(x_new):
            # Fallback: small step
            x_new = x + 1e-3
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
        else:
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                # Shrink step
                x_new = x + 0.5 * (x - x0)
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not math.isfinite(f_new):
                    break

        # Safeguard: Limit step size
        max_step = max(1.0, abs(x) * 0.5)
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break

        # Line search / Acceptance
        # Accept if |f_new| < |f|
        if abs(f_new) < abs(fx):
            # Update best
            if abs(f_new) < best_abs_fx:
                best_x = x_new
                best_fx = f_new
                best_abs_fx = abs(f_new)

            if abs(f_new) <= 1e-8:
                return x_new

            # Move to new point
            # Add to history, keeping last 2 for Secant
            history.append((x_new, f_new))
            if len(history) > 3:
                history.pop(0)

            x = x_new
            fx = f_new
        else:
            # Backtrack or try different strategy
            # If Newton/Secant failed, try bisection-like step towards a point where sign changes?
            # We don't have a bracket.
            # Let's try a smaller step in the same direction.
            x_try = x + 0.5 * (x_new - x)
            try:
                f_try = f(x_try)
            except Exception:
                break
            if math.isfinite(f_try) and abs(f_try) < abs(fx):
                if abs(f_try) < best_abs_fx:
                    best_x = x_try
                    best_fx = f_try
                    best_abs_fx = abs(f_try)
                if abs(f_try) <= 1e-8:
                    return x_try
                history.append((x_try, f_try))
                if len(history) > 3:
                    history.pop(0)
                x = x_try
                fx = f_try
            else:
                # Stuck. Try a random small perturbation? Or break.
                # Let's try a few random nearby points to escape local minima of |f|^2
                # But for polynomials, local minima of |f|^2 are either roots or points where f'=0 and f!=0.
                # If we are at a point where f'=0, Newton fails. Secant might work.
                # If Secant also fails, we are in trouble.
                # Let's just break and return best.
                break

    if abs(best_fx) <= 1e-8:
        return best_x

    return best_x
# EVOLVE-BLOCK-END
