"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def find_root(f, x0):
    # Deadline guard: evaluator kills at 150s wall-clock per evaluation.
    # We use a conservative 145s deadline to be safe.
    deadline = time.time() + 145.0
    eval_count = 0
    max_evals = 500

    def safe_f(x):
        nonlocal eval_count
        if time.time() > deadline:
            raise TimeoutError()
        if eval_count >= max_evals:
            raise ValueError("max evals exceeded")
        eval_count += 1
        try:
            val = f(x)
            val = float(val)
        except Exception:
            return float('nan')
        return val

    x = float(x0)
    best_x = x
    best_fabs = float('inf')

    # Track the best (smallest |f(x)|) we've seen
    def update_best(x_val, fx_val):
        nonlocal best_x, best_fabs
        if math.isfinite(fx_val) and abs(fx_val) < best_fabs:
            best_fabs = abs(fx_val)
            best_x = x_val

    # Initial evaluation
    try:
        fx = safe_f(x)
        update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x
    except (TimeoutError, ValueError):
        return best_x

    # Strategy: Use a robust root-finding approach combining:
    # 1. Bisection-like bracketing with adaptive steps
    # 2. Newton's method when derivative is available and stable
    # 3. Secant method as fallback
    # 4. Final high-precision polish

    # First, try to find a bracket by expanding from x0
    bracket_lo = None
    bracket_hi = None
    fx_lo = None
    fx_hi = None

    # Check current point
    if math.isfinite(fx):
        if fx > 0:
            bracket_hi = x
            fx_hi = fx
        else:
            bracket_lo = x
            fx_lo = fx

    # Try to find a sign change by expanding
    step = 1.0
    for _ in range(30):
        if time.time() > deadline:
            break
        # Try x + step
        x_try = x + step
        try:
            fx_try = safe_f(x_try)
            update_best(x_try, fx_try)
            if math.isfinite(fx_try):
                if fx_try > 0:
                    if bracket_lo is not None:
                        bracket_hi = x_try
                        fx_hi = fx_try
                        break
                else:
                    if bracket_hi is not None:
                        bracket_lo = x_try
                        fx_lo = fx_try
                        break
        except (TimeoutError, ValueError):
            break

        # Try x - step
        x_try = x - step
        try:
            fx_try = safe_f(x_try)
            update_best(x_try, fx_try)
            if math.isfinite(fx_try):
                if fx_try > 0:
                    if bracket_lo is not None:
                        bracket_hi = x_try
                        fx_hi = fx_try
                        break
                else:
                    if bracket_hi is not None:
                        bracket_lo = x_try
                        fx_lo = fx_try
                        break
        except (TimeoutError, ValueError):
            break

        step *= 2.0

    # If we have a bracket, use bisection + Newton hybrid
    if bracket_lo is not None and bracket_hi is not None:
        lo = bracket_lo
        hi = bracket_hi
        f_lo = fx_lo
        f_hi = fx_hi

        for _ in range(200):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2.0
            try:
                f_mid = safe_f(mid)
                update_best(mid, f_mid)
            except (TimeoutError, ValueError):
                break

            if not math.isfinite(f_mid):
                # If mid gives nan/inf, shrink bracket
                if mid > lo:
                    hi = mid
                else:
                    lo = mid
                continue

            if abs(f_mid) <= 1e-8:
                return mid

            # Determine which sub-interval contains a sign change
            if f_mid * f_lo <= 0:
                hi = mid
                f_hi = f_mid
            else:
                lo = mid
                f_lo = f_mid

            # Newton step if possible (use secant-like update)
            if hi - lo < 1e-15:
                break

            # Try a Newton-like step using the two bracket endpoints
            # as a secant approximation
            if abs(f_hi - f_lo) > 1e-300:
                x_new = lo - f_lo * (hi - lo) / (f_hi - f_lo)
                # Only accept if within bracket
                if lo < x_new < hi:
                    try:
                        f_new = safe_f(x_new)
                        update_best(x_new, f_new)
                        if math.isfinite(f_new):
                            if abs(f_new) <= 1e-8:
                                return x_new
                            if f_new * f_lo <= 0:
                                hi = x_new
                                f_hi = f_new
                            else:
                                lo = x_new
                                f_lo = f_new
                    except (TimeoutError, ValueError):
                        break

        # Final check
        if abs(f_mid) <= 1e-8 if 'f_mid' in dir() else False:
            return mid
        # Return the best we found
        return best_x

    # No bracket found; use Newton/secant method with damping
    # Use secant method with two initial points
    x1 = x
    fx1 = fx
    h = 1e-6 * max(1.0, abs(x))
    x2 = x + h
    try:
        fx2 = safe_f(x2)
        update_best(x2, fx2)
    except (TimeoutError, ValueError):
        return best_x

    if not math.isfinite(fx2):
        # Try different perturbation
        x2 = x - h
        try:
            fx2 = safe_f(x2)
            update_best(x2, fx2)
        except (TimeoutError, ValueError):
            return best_x

    if not math.isfinite(fx2) or abs(fx2 - fx1) < 1e-300:
        # Fallback: simple damped Newton with finite difference
        for _ in range(100):
            if time.time() > deadline:
                break
            h = 1e-6 * max(1.0, abs(x1))
            try:
                d = (safe_f(x1 + h) - fx1) / h
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(d) or abs(d) < 1e-300:
                break
            step_new = fx1 / d
            # Damp the step
            x_new = x1 - step_new
            # Limit step size
            if abs(x_new - x1) > 10.0:
                x_new = x1 + 10.0 * math.copysign(1.0, -step_new)
            try:
                fx_new = safe_f(x_new)
                update_best(x_new, fx_new)
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(fx_new):
                break
            if abs(fx_new) <= 1e-8:
                return x_new
            x1 = x_new
            fx1 = fx_new
        return best_x

    # Secant method
    for _ in range(200):
        if time.time() > deadline:
            break
        if abs(fx2 - fx1) < 1e-300:
            break
        x_new = x1 - fx1 * (x2 - x1) / (fx2 - fx1)
        # Limit step
        if abs(x_new - x1) > 100.0 * max(1.0, abs(x1)):
            x_new = x1 + 100.0 * max(1.0, abs(x1)) * math.copysign(1.0, x2 - x1)
        try:
            fx_new = safe_f(x_new)
            update_best(x_new, fx_new)
        except (TimeoutError, ValueError):
            break
        if not math.isfinite(fx_new):
            # Perturb and retry
            x2 = x1 + 1e-6 * max(1.0, abs(x1))
            try:
                fx2 = safe_f(x2)
                update_best(x2, fx2)
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(fx2):
                break
            continue
        if abs(fx_new) <= 1e-8:
            return x_new
        x1 = x2
        fx1 = fx2
        x2 = x_new
        fx2 = fx_new

    # Final polish: if we're close, try Newton with high precision
    if best_fabs <= 1e-4:
        x_polish = best_x
        for _ in range(50):
            if time.time() > deadline:
                break
            try:
                fx_p = safe_f(x_polish)
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(fx_p):
                break
            if abs(fx_p) <= 1e-8:
                return x_polish
            h = 1e-8 * max(1.0, abs(x_polish))
            try:
                d = (safe_f(x_polish + h) - fx_p) / h
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(d) or abs(d) < 1e-300:
                break
            x_new = x_polish - fx_p / d
            if abs(x_new - x_polish) < 1e-15:
                break
            try:
                fx_new = safe_f(x_new)
                update_best(x_new, fx_new)
            except (TimeoutError, ValueError):
                break
            if not math.isfinite(fx_new):
                break
            x_polish = x_new

    return best_x
# EVOLVE-BLOCK-END
