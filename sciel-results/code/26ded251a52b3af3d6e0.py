import math
import time
from scipy.optimize import brentq, newton

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    call_count = 0
    max_calls = 490
    
    def eval_f(x_val):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        call_count += 1
        return f(x_val)
    
    x = float(x0)
    
    # Initial evaluation
    try:
        fx = eval_f(x)
    except Exception:
        return x
    
    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    # Adaptive search for a bracket
    # We scan in both directions with exponentially growing step sizes
    # This is more efficient than a fixed grid for unknown periods
    
    bracket_found = False
    a_bracket = None
    b_bracket = None
    
    # Start with a small step and grow it
    # We alternate between left and right to be fair
    step = 1e-4
    max_step = 1e8
    
    prev_x = x
    prev_f = fx
    
    while step < max_step and call_count < max_calls and time.time() < deadline:
        # Try right
        x_right = x + step
        try:
            f_right = eval_f(x_right)
        except Exception:
            f_right = None
        
        if f_right is not None and math.isfinite(f_right):
            if abs(f_right) < best_abs:
                best_x = x_right
                best_f = f_right
                best_abs = abs(f_right)
            if best_abs <= 1e-8:
                return best_x
            
            # Check for sign change between x and x_right
            if prev_f * f_right < 0:
                a_bracket, b_bracket = prev_x, x_right
                bracket_found = True
                break
            elif prev_f * f_right > 0 and step > 1.0:
                # If we've taken a large step and no sign change, 
                # we might have skipped a root or the function is flat.
                # For oscillatory functions, a large step without sign change 
                # is suspicious, but we keep growing to be safe.
                pass
        
        # Try left
        x_left = x - step
        try:
            f_left = eval_f(x_left)
        except Exception:
            f_left = None
        
        if f_left is not None and math.isfinite(f_left):
            if abs(f_left) < best_abs:
                best_x = x_left
                best_f = f_left
                best_abs = abs(f_left)
            if best_abs <= 1e-8:
                return best_x
            
            if prev_f * f_left < 0:
                a_bracket, b_bracket = x_left, prev_x
                bracket_found = True
                break
        
        step *= 2.0
    
    if bracket_found:
        try:
            root = brentq(f, a_bracket, b_bracket, xtol=1e-14, rtol=1e-14, maxiter=100)
            if time.time() < deadline:
                # Verify
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # Fallback: manual bisection if brentq failed
        a_b, b_b = min(a_bracket, b_bracket), max(a_bracket, b_bracket)
        fa_b, fb_b = f(a_b), f(b_b)
        if fa_b * fb_b < 0:
            for _ in range(60):
                if time.time() > deadline:
                    break
                m = (a_b + b_b) / 2.0
                fm = f(m)
                if abs(fm) <= 1e-8:
                    return m
                if fa_b * fm < 0:
                    b_b = m
                    fb_b = fm
                else:
                    a_b = m
                    fa_b = fm
            return (a_b + b_b) / 2.0
        else:
            return best_x
    
    # No bracket found. Try Newton's method from the best point.
    # This is useful if the function is smooth and we are close to a root.
    x_curr = best_x
    f_curr = best_f
    h = 1e-5 * max(1.0, abs(x_curr))
    
    for _ in range(30):
        if time.time() > deadline or call_count >= max_calls:
            break
        try:
            fp = eval_f(x_curr + h)
            fm = eval_f(x_curr - h)
        except Exception:
            break
        
        if not (math.isfinite(fp) and math.isfinite(fm)):
            break
            
        deriv = (fp - fm) / (2 * h)
        if deriv == 0 or not math.isfinite(deriv):
            break
            
        x_new = x_curr - f_curr / deriv
        if not math.isfinite(x_new):
            break
            
        # Damping
        step_size = abs(x_new - x_curr)
        if step_size > 10.0 * max(1.0, abs(x_curr)):
            x_new = x_curr + math.copysign(10.0 * max(1.0, abs(x_curr)), x_new - x_curr)
            
        try:
            f_new = eval_f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) < best_abs:
            best_x = x_new
            best_f = f_new
            best_abs = abs(f_new)
            if best_abs <= 1e-8:
                return best_x
            x_curr = x_new
            f_curr = f_new
            h *= 0.5
        else:
            break
            
    return best_x
