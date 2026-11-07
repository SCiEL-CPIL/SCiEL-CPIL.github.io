import math
import time

def find_root(f, x0):
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    def time_ok():
        return time.time() - start_time < DEADLINE
    
    # Check initial point
    f0 = f(x0)
    if not math.isfinite(f0):
        # Try nearby points
        for dx in [1e-3, -1e-3, 1e-2, -1e-2, 0.1, -0.1, 1.0, -1.0]:
            if not time_ok():
                return x0
            x_try = x0 + dx
            ft = f(x_try)
            if math.isfinite(ft) and abs(ft) <= 1e-8:
                return x_try
        return x0
    
    if abs(f0) <= 1e-8:
        return x0
    
    # Try Newton's method first (2 calls per step)
    x = x0
    fx = f0
    newton_success = False
    for i in range(30):
        if not time_ok():
            break
        h = 1e-7 * max(1.0, abs(x))
        fx2 = f(x + h)
        if not math.isfinite(fx2):
            break
        d = (fx2 - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        # Damping if step is too large
        if abs(x_new - x) > 10 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
        x = x_new
        fx = f(x)
        if not math.isfinite(fx):
            break
        if abs(fx) <= 1e-10:
            newton_success = True
            break
    
    if newton_success and abs(fx) <= 1e-8:
        return x
    
    # Newton failed - do adaptive scanning for brackets
    # Try multiple scales for oscillatory functions
    best_bracket = None
    best_bracket_width = float('inf')
    
    # Scan outward from x0 with various step sizes
    # For oscillatory functions, try periods of ~0.1 to ~10
    base_steps = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    for step in base_steps:
        if not time_ok():
            break
        # Scan in both directions
        for direction in [1, -1]:
            if not time_ok():
                break
            prev_x = x0
            prev_f = f0
            for i in range(1, 50):
                if not time_ok():
                    break
                x_curr = x0 + direction * i * step
                f_curr = f(x_curr)
                if not math.isfinite(f_curr):
                    break
                if abs(f_curr) <= 1e-8:
                    return x_curr
                # Check for sign change
                if prev_f * f_curr < 0:
                    width = abs(x_curr - prev_x)
                    if width < best_bracket_width:
                        best_bracket_width = width
                        best_bracket = (prev_x, x_curr, prev_f, f_curr)
                prev_x = x_curr
                prev_f = f_curr
                # Early exit if we found a good bracket
                if best_bracket and best_bracket_width < step * 0.5:
                    break
    
    # If we found a bracket, use bisection + Newton hybrid
    if best_bracket:
        a, b, fa, fb = best_bracket
        if fa * fb > 0:
            return x0  # fallback
        
        # Use a robust root finder: bisection with Newton acceleration
        x_lo = a
        x_hi = b
        f_lo = fa
        f_hi = fb
        
        for i in range(100):
            if not time_ok():
                break
            # Bisection point
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid = f(x_mid)
            if not math.isfinite(f_mid):
                break
            if abs(f_mid) <= 1e-8:
                return x_mid
            
            # Try Newton step
            h = 1e-7 * max(1.0, abs(x_mid))
            f_mid2 = f(x_mid + h)
            if math.isfinite(f_mid2):
                d = (f_mid2 - f_mid) / h
                if d != 0 and math.isfinite(d):
                    x_newton = x_mid - f_mid / d
                    # Check if Newton step stays in bracket
                    if x_lo < x_newton < x_hi:
                        x_mid = x_newton
                        f_mid = f(x_mid)
                        if not math.isfinite(f_mid):
                            break
                        if abs(f_mid) <= 1e-8:
                            return x_mid
            
            # Update bracket
            if f_lo * f_mid <= 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid
            
            # Convergence check
            if abs(x_hi - x_lo) < 1e-12:
                break
        
        x_result = 0.5 * (x_lo + x_hi)
        f_result = f(x_result)
        if math.isfinite(f_result) and abs(f_result) <= 1e-8:
            return x_result
    
    # Fallback: return the best x we've seen
    return x0
