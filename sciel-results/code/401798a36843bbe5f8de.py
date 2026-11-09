"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    deadline = time.time() + 140.0
    x = float(x0)
    
    if not math.isfinite(x):
        return x0
    
    best_x = x
    best_fx = f(x)
    best_abs = abs(best_fx)
    
    if not math.isfinite(best_fx):
        return x0
    
    if best_abs <= 1e-8:
        return x
    
    # Phase 1: Try to establish a bracket
    h = 1e-4 * max(1.0, abs(x))
    a, fa = x, best_fx
    b = x + h
    fb = f(b)
    
    if not math.isfinite(fb):
        return best_x if best_abs <= 1e-8 else x
    
    if abs(fb) < best_abs:
        best_x = b
        best_abs = abs(fb)
    
    # Expand bracket if needed
    if fa * fb > 0:
        # Same sign, expand
        for _ in range(50):
            h *= 2.0
            b = x + h
            fb = f(b)
            if not math.isfinite(fb):
                b = x - h
                fb = f(b)
                if not math.isfinite(fb):
                    break
            if abs(fb) < best_abs:
                best_x = b
                best_abs = abs(fb)
            if fa * fb <= 0:
                break
        else:
            # No bracket found, fall back to secant
            pass
    
    if fa * fb <= 0 and math.isfinite(fb):
        # We have a bracket [a, b] with f(a)*f(b) <= 0
        # Use Illinois method for robust convergence
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        for i in range(60):
            if time.time() > deadline:
                break
            if abs(fb) <= 1e-8:
                return b
            if abs(fa) <= 1e-8:
                return a
            
            # Illinois: use linear interpolation, but if same side as previous, use bisection
            denom = fb - fa
            if abs(denom) < 1e-30:
                # Degenerate, bisect
                b = (a + b) / 2.0
            else:
                # Linear interpolation point
                b_new = b - fb * (b - a) / denom
                # Check if b_new is on the same side as b relative to midpoint
                mid = (a + b) / 2.0
                if (b_new - mid) * (b - mid) > 0:
                    # Same side, use bisection
                    b_new = mid
                else:
                    # Different side, use interpolation
                    # But keep it within [a, b]
                    if b_new < a:
                        b_new = a + (b - a) * 0.25
                    elif b_new > b:
                        b_new = b - (b - a) * 0.25
            
            fb = f(b_new)
            if not math.isfinite(fb):
                b_new = (a + b) / 2.0
                fb = f(b_new)
                if not math.isfinite(fb):
                    break
            
            if abs(fb) < best_abs:
                best_x = b_new
                best_abs = abs(fb)
            
            if abs(fb) <= 1e-8:
                return b_new
            
            if fa * fb <= 0:
                # Root is in [a, b_new]
                b = b_new
            else:
                # Root is in [b_new, b]
                a, fa = b_new, fb
            
            # If interval is very small, we're done
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        
        if abs(fb) <= 1e-8:
            return b
        if abs(fa) <= 1e-8:
            return a
        
        # Return the better of a or b
        if abs(fa) < abs(fb):
            return a
        else:
            return b
    
    else:
        # No bracket found, use secant method
        x_prev = a
        f_prev = fa
        x_curr = b
        f_curr = fb
        
        for i in range(50):
            if time.time() > deadline:
                break
            if abs(f_curr) <= 1e-8:
                return x_curr
            if abs(f_curr) < best_abs:
                best_x = x_curr
                best_abs = abs(f_curr)
            
            if abs(f_curr - f_prev) < 1e-30:
                break
            
            x_next = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            if not math.isfinite(x_next):
                break
            
            # Damping
            step = abs(x_next - x_curr)
            if step > 10.0 * max(1.0, abs(x_curr)):
                x_next = x_curr + math.copysign(10.0 * max(1.0, abs(x_curr)), x_next - x_curr)
            
            x_prev, f_prev = x_curr, f_curr
            x_curr, f_curr = x_next, f(x_next)
            
            if not math.isfinite(f_curr):
                x_curr, f_curr = x_prev, f_prev
                break
        
        if abs(f_curr) <= 1e-8:
            return x_curr
    
    # Final: return best
    if best_abs <= 1e-8:
        return best_x
    
    # One more try with current best
    if math.isfinite(best_x):
        return best_x
    return x0
# EVOLVE-BLOCK-END
