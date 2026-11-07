import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 490  # Leave margin for safety
    DEADLINE = time.time() + 145  # 145s out of 150s limit
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            abs_val = abs(val)
            if abs_val < best_abs:
                best_abs = abs_val
                best_x = x
            return val
        except Exception:
            return float('inf')
    
    x = float(x0)
    fx = safe_f(x)
    if abs(fx) <= TOL:
        return x
    
    # Try to bracket the root by scanning outward with adaptive step sizes
    # Use a multi-scale approach: try several step magnitudes
    found_bracket = False
    bracket_a = x
    bracket_b = None
    
    # For oscillatory functions, the root might be nearby or at a characteristic scale
    # Try a sequence of step sizes from small to large
    base_scale = max(1.0, abs(x))
    step_multipliers = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    
    for mult in step_multipliers:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        step = mult * base_scale
        # Scan right
        x_curr = float(x0)
        f_curr = safe_f(x_curr) if x_curr != x else fx
        for i in range(20):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_try = x_curr + step
            f_try = safe_f(x_try)
            if math.isfinite(f_try):
                if f_curr * f_try < 0:
                    bracket_a = x_curr
                    bracket_b = x_try
                    found_bracket = True
                    break
                f_curr = f_try
                x_curr = x_try
            else:
                break
        if found_bracket:
            break
        # Scan left
        x_curr = float(x0)
        f_curr = safe_f(x_curr) if x_curr != x else fx
        for i in range(20):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            x_try = x_curr - step
            f_try = safe_f(x_try)
            if math.isfinite(f_try):
                if f_curr * f_try < 0:
                    bracket_a = x_try
                    bracket_b = x_curr
                    found_bracket = True
                    break
                f_curr = f_try
                x_curr = x_try
            else:
                break
        if found_bracket:
            break
    
    if found_bracket:
        try:
            result = brentq(f, bracket_a, bracket_b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if math.isfinite(result):
                # Final polish: verify and potentially refine
                fx_result = safe_f(result)
                if abs(fx_result) <= TOL:
                    return result
                # If brentq didn't get us close enough, try to polish with Newton
                x = result
                for _ in range(5):
                    if calls >= MAX_CALLS or time.time() > DEADLINE:
                        break
                    fx = safe_f(x)
                    if abs(fx) <= TOL:
                        return x
                    h = 1e-8 * max(1.0, abs(x))
                    if calls >= MAX_CALLS:
                        break
                    fx_plus = safe_f(x + h)
                    if not math.isfinite(fx_plus):
                        break
                    d = (fx_plus - fx) / h
                    if d == 0 or not math.isfinite(d):
                        break
                    step_new = fx / d
                    if abs(step_new) > 1e-4 * max(1.0, abs(x)):
                        step_new = 1e-4 * max(1.0, abs(x)) * math.copysign(1, step_new)
                    x_new = x - step_new
                    if not math.isfinite(x_new):
                        break
                    x = x_new
                if abs(safe_f(x)) <= TOL:
                    return x
        except Exception:
            pass
    
    # Fallback: try Newton's method with damped steps
    x = float(x0)
    for _ in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        h = 1e-6 * max(1.0, abs(x))
        if calls >= MAX_CALLS:
            break
        fx_plus = safe_f(x + h)
        if not math.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        step = fx / d
        # Damp the step to avoid divergence
        if abs(step) > 10 * max(1.0, abs(x)):
            step = 10 * max(1.0, abs(x)) * math.copysign(1, step)
        x_new = x - step
        if not math.isfinite(x_new):
            break
        x = x_new
    
    if calls >= MAX_CALLS or time.time() > DEADLINE:
        return best_x
    
    # Final check
    fx_best = safe_f(best_x)
    if abs(fx_best) <= TOL:
        return best_x
    
    return best_x
