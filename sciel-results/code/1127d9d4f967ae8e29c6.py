"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def _check_deadline(deadline):
    return time.time() < deadline

def find_root(f, x0):
    start_time = time.time()
    # 150s total limit for 3 equations. 10s per equation is safe.
    deadline = start_time + 10.0
    
    x = float(x0)
    best_x = x
    try:
        best_f = f(x)
    except Exception:
        return x

    if not math.isfinite(best_f):
        return x

    if abs(best_f) <= 1e-8:
        return best_x

    # Track previous point for secant method
    x_prev = x
    f_prev = best_f
    x = x + 1e-4 * max(1.0, abs(x))
    
    try:
        f_curr = f(x)
    except Exception:
        f_curr = best_f
        x = x_prev

    if not math.isfinite(f_curr):
        x = x_prev
        f_curr = best_f

    if abs(f_curr) < abs(best_f):
        best_f = f_curr
        best_x = x

    if abs(f_curr) <= 1e-12:
        return x

    # Main loop: use secant method (2 calls per iteration) with Newton polish
    for _ in range(200):
        if not _check_deadline(deadline):
            break

        try:
            fx = f(x)
        except Exception:
            break
            
        if not math.isfinite(fx):
            break
            
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
            
        if abs(fx) <= 1e-12:
            return x

        # Try Newton step with central difference if we have enough budget
        # Otherwise use secant
        if _check_deadline(deadline):
            h = 1e-6 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
                fx_minus = f(x - h)
            except Exception:
                fx_plus = None
                fx_minus = None
                
            if fx_plus is not None and fx_minus is not None and math.isfinite(fx_plus) and math.isfinite(fx_minus):
                dfx = (fx_plus - fx_minus) / (2.0 * h)
                if dfx != 0.0:
                    x_new = x - fx / dfx
                    if math.isfinite(x_new):
                        # Safeguard: Limit step size
                        max_step = 0.5 * (1.0 + abs(x))
                        step = x_new - x
                        if abs(step) > max_step:
                            x_new = x + math.copysign(max_step, step)
                        x_prev = x
                        f_prev = fx
                        x = x_new
                        continue
            # Fall through to secant if Newton fails

        # Secant method
        if f_curr == f_prev:
            break
        if f_curr - f_prev == 0.0:
            break
            
        x_new = x - fx * (x - x_prev) / (fx - f_prev)
        
        if not math.isfinite(x_new):
            break
            
        # Safeguard: Limit step size
        max_step = 1.0 + abs(x)
        step = x_new - x
        if abs(step) > max_step:
            x_new = x + math.copysign(max_step, step)
            
        x_prev = x
        f_prev = fx
        x = x_new
        f_curr = fx

    # Final polish: try a few Newton steps with tight tolerance
    for _ in range(5):
        if not _check_deadline(deadline):
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not math.isfinite(fx):
            break
        if abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        if abs(fx) <= 1e-12:
            return x
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
        except Exception:
            break
        if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
            break
        dfx = (fx_plus - fx_minus) / (2.0 * h)
        if dfx == 0.0:
            break
        x_new = x - fx / dfx
        if not math.isfinite(x_new):
            break
        x = x_new

    return best_x
# EVOLVE-BLOCK-END
