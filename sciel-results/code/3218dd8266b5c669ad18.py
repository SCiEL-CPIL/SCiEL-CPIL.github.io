"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x = float(x0)
    deadline = time.monotonic() + 115.0
    best_x = x
    best_fx = None
    call_count = 0

    def safe_f(val):
        nonlocal call_count
        try:
            result = f(val)
            call_count += 1
            if result is not None and math.isfinite(result):
                return float(result)
        except Exception:
            pass
        return None

    def is_success(fx_val):
        return fx_val is not None and abs(fx_val) <= 1e-8

    # Evaluate at starting point
    best_fx = safe_f(x)
    if is_success(best_fx):
        return x

    # Try Newton's method with adaptive step
    x_newton = x
    fx = best_fx
    for _ in range(40):
        if time.monotonic() > deadline:
            break
        if fx is None:
            break
        if abs(fx) <= 1e-10:
            return x_newton
        
        h = 1e-6 * max(1.0, abs(x_newton))
        fx_plus = safe_f(x_newton + h)
        if fx_plus is None:
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        step = fx / d
        # Damp the step if it's too large
        if abs(step) > 10 * max(1.0, abs(x_newton)):
            step = math.copysign(10 * max(1.0, abs(x_newton)), step)
        
        x_newton = x_newton - step
        if not math.isfinite(x_newton):
            break
        
        fx = safe_f(x_newton)
        if fx is None:
            break
        
        if best_fx is None or abs(fx) < abs(best_fx):
            best_x = x_newton
            best_fx = fx
        
        if is_success(fx):
            return x_newton

    if is_success(best_fx):
        return best_x

    # Secant method (uses two previous points, no derivative needed)
    # Try to find a good second point
    scale = max(1.0, abs(x0))
    x1 = x0 + scale * 0.1
    fx1 = safe_f(x1)
    
    if fx1 is not None and best_fx is not None:
        x0_sec = x
        x1_sec = x1
        fx0 = best_fx
        fx1 = safe_f(x1)  # re-evaluate to be safe
        
        for _ in range(30):
            if time.monotonic() > deadline:
                break
            if fx1 is None or fx0 is None:
                break
            if abs(fx1) <= 1e-10:
                return x1_sec
            
            denom = fx1 - fx0
            if denom == 0 or not math.isfinite(denom):
                break
            
            x2 = x1_sec - fx1 * (x1_sec - x0_sec) / denom
            if not math.isfinite(x2):
                break
            
            fx2 = safe_f(x2)
            if fx2 is None:
                break
            
            if best_fx is None or abs(fx2) < abs(best_fx):
                best_x = x2
                best_fx = fx2
            
            if is_success(fx2):
                return x2
            
            x0_sec = x1_sec
            fx0 = fx1
            x1_sec = x2
            fx1 = fx2

    if is_success(best_fx):
        return best_x

    # Bisection with adaptive bracketing
    scale = max(1.0, abs(x0))
    left = x0 - scale
    right = x0 + scale
    f_left = None
    f_right = None

    # Try to find a sign change
    for expand_factor in [1, 2, 4, 8, 16, 32, 64, 128]:
        if time.monotonic() > deadline:
            break
        left = x0 - scale * expand_factor
        right = x0 + scale * expand_factor
        f_left = safe_f(left)
        f_right = safe_f(right)
        
        if f_left is None or f_right is None:
            continue
        if f_left * f_right <= 0:
            break
    
    # If we found a sign change, do bisection
    if f_left is not None and f_right is not None and f_left * f_right <= 0:
        for _ in range(80):
            if time.monotonic() > deadline:
                break
            mid = (left + right) / 2.0
            f_mid = safe_f(mid)
            if f_mid is None:
                break
            if abs(f_mid) <= 1e-10:
                return mid
            if best_fx is None or abs(f_mid) < abs(best_fx):
                best_x = mid
                best_fx = f_mid
            if is_success(f_mid):
                return mid
            if f_mid * f_left <= 0:
                right = mid
                f_right = f_mid
            else:
                left = mid
                f_left = f_mid

    if is_success(best_fx):
        return best_x
    
    # Last resort: try a few random-ish points near x0
    for offset in [0.01, -0.01, 0.1, -0.1, 1.0, -1.0]:
        if time.monotonic() > deadline:
            break
        x_try = x0 + offset * scale
        fx_try = safe_f(x_try)
        if fx_try is not None:
            if best_fx is None or abs(fx_try) < abs(best_fx):
                best_x = x_try
                best_fx = fx_try
            if is_success(fx_try):
                return x_try

    return best_x
# EVOLVE-BLOCK-END
