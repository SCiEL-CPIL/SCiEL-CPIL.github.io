import math
import time
from scipy.optimize import brentq

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    call_count = 0
    max_calls = 490  # Leave some buffer for safety
    
    x = float(x0)
    
    def eval_f(x_val):
        nonlocal call_count
        if call_count >= max_calls:
            raise RuntimeError("Max calls exceeded")
        if time.time() > deadline:
            raise RuntimeError("Time limit exceeded")
        call_count += 1
        return f(x_val)

    # Initial check
    try:
        fx = eval_f(x)
    except Exception:
        return x  # Return x0 if initial eval fails

    if not math.isfinite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x

    # Strategy:
    # 1. Try to find a bracket [a, b] where f(a)*f(b) < 0.
    # 2. If found, use brentq.
    # 3. If not, find the point with minimum |f| and try to refine it using Newton's method or local search.
    
    best_x = x
    best_f = fx
    best_abs = abs(fx)
    
    # Logarithmic scale scan to efficiently cover a wide range of step sizes
    # This avoids wasting calls on fixed-step scans for functions with unknown periods
    # Start with small steps (in case root is close) and expand logarithmically
    scales = []
    for exp in range(-6, 7):  # 1e-6 to 1e6
        for mult in [1.0, 2.0, 5.0]:
            scales.append(mult * (10.0 ** exp))
    
    bracket_found = False
    lo = 0.0
    hi = 0.0
    
    for scale in scales:
        if time.time() > deadline or call_count >= max_calls:
            break
            
        # Check right
        x_r = x + scale
        try:
            fr = eval_f(x_r)
        except Exception:
            fr = None
        
        if fr is not None and math.isfinite(fr):
            abs_fr = abs(fr)
            if abs_fr < best_abs:
                best_abs = abs_fr
                best_x = x_r
                best_f = fr
            if abs_fr <= 1e-8:
                return x_r
            if fx * fr < 0:
                lo, hi = x, x_r
                bracket_found = True
                break
        else:
            pass

        # Check left
        x_l = x - scale
        try:
            fl = eval_f(x_l)
        except Exception:
            fl = None
            
        if fl is not None and math.isfinite(fl):
            abs_fl = abs(fl)
            if abs_fl < best_abs:
                best_abs = abs_fl
                best_x = x_l
                best_f = fl
            if abs_fl <= 1e-8:
                return x_l
            if fx * fl < 0:
                lo, hi = x_l, x
                bracket_found = True
                break

    if bracket_found:
        # Use Brent's method with tighter tolerances
        try:
            root = brentq(f, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=200)
            if time.time() < deadline:
                # Verify
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # If brentq failed or wasn't good enough, try bisection manually
        a_b, b_b = min(lo, hi), max(lo, hi)
        fa_b, fb_b = f(a_b), f(b_b)
        if fa_b * fb_b < 0:
            for _ in range(80):
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
            m = (a_b + b_b) / 2.0
            return m

    # If no bracket found, refine the best point found
    if best_abs > 1e-8:
        # Try to refine best_x using Newton's method with central difference
        h = 1e-5 * max(1.0, abs(best_x))
        
        for _ in range(20):
            if time.time() > deadline or call_count >= max_calls:
                break
            
            try:
                fx_plus = eval_f(best_x + h)
                fx_minus = eval_f(best_x - h)
            except Exception:
                break
                
            if not (math.isfinite(fx_plus) and math.isfinite(fx_minus)):
                break
                
            deriv = (fx_plus - fx_minus) / (2.0 * h)
            
            if deriv == 0 or not math.isfinite(deriv):
                break
                
            # Newton step
            x_new = best_x - best_f / deriv
            
            if not math.isfinite(x_new):
                break
                
            # Damping: if the step is too large, damp it to avoid diverging
            step_size = abs(x_new - best_x)
            if step_size > 10.0 * max(1.0, abs(best_x)):
                x_new = best_x + math.copysign(10.0 * max(1.0, abs(best_x)), x_new - best_x)
                
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
            else:
                break
            h *= 0.5

    # Final check
    if not math.isfinite(best_x):
        return x0
    return best_x
