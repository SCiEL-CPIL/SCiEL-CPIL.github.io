"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    deadline = time.time() + 140  # 140s safety margin (evaluator kills at 150s)
    eval_count = 0
    max_evals = 500
    best_x = float(x0)
    best_fval = float('inf')
    
    def safe_f(x):
        nonlocal eval_count, best_x, best_fval
        if time.time() > deadline:
            raise TimeoutError()
        eval_count += 1
        if eval_count > max_evals:
            raise ValueError("max evals")
        val = f(x)
        val = float(val)
        if math.isfinite(val):
            if abs(val) < abs(best_fval):
                best_fval = val
                best_x = x
        return val
    
    x = float(x0)
    
    # Try to get initial points
    f0 = safe_f(x)
    if abs(f0) <= 1e-8:
        return x
    
    # Get a second point for secant
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = safe_f(x1)
    if abs(f1) <= 1e-8:
        return x1
    
    # Maintain three points for inverse quadratic interpolation
    # (x0, f0), (x1, f1), and we'll get x2
    # Use secant first to get x2
    if f1 != f0:
        x2 = x1 - f1 * (x1 - x) / (f1 - f0)
    else:
        x2 = x1 + h
    
    f2 = safe_f(x2)
    if abs(f2) <= 1e-8:
        return x2
    
    # Now we have three points: (x, f0), (x1, f1), (x2, f2)
    # Use a hybrid approach: try IQI, but fall back to secant if IQI step is bad
    
    for _ in range(490):  # We've used 3 evals, have 497 left
        if time.time() > deadline:
            break
        
        # Sort the three points by |f| value to use the best three
        points = [(x, f0), (x1, f1), (x2, f2)]
        points.sort(key=lambda p: abs(p[1]))
        
        # Check if we already have a good enough root
        if abs(points[0][1]) <= 1e-8:
            return points[0][0]
        
        # Use the three points with smallest |f|
        xa, fa = points[0]
        xb, fb = points[1]
        xc, fc = points[2]
        
        # First, try secant with the two best points
        if fa != fb:
            x_secant = xb - fb * (xb - xa) / (fb - fa)
        else:
            x_secant = (xa + xb) / 2
        
        # Try IQI if possible
        x_iqi = None
        try:
            if abs(fa) > 1e-15 and abs(fb) > 1e-15 and abs(fc) > 1e-15:
                denom_a = (fa - fb) * (fa - fc)
                denom_b = (fb - fa) * (fb - fc)
                denom_c = (fc - fa) * (fc - fb)
                
                if abs(denom_a) > 1e-30 and abs(denom_b) > 1e-30 and abs(denom_c) > 1e-30:
                    x_iqi = (xa * fb * fc / denom_a + 
                             xb * fa * fc / denom_b + 
                             xc * fa * fb / denom_c)
        except (ZeroDivisionError, OverflowError):
            pass
        
        # Choose between secant and IQI
        # Prefer IQI if it's in the "reasonable" range (not too far from current best)
        if x_iqi is not None and math.isfinite(x_iqi):
            # Check if IQI step is reasonable (within a factor of 10 of secant step magnitude)
            step_secant = abs(x_secant - xa)
            step_iqi = abs(x_iqi - xa)
            if step_secant > 0:
                ratio = step_iqi / step_secant
                if 0.1 < ratio < 10.0:
                    x_new = x_iqi
                else:
                    x_new = x_secant
            else:
                x_new = x_iqi
        else:
            x_new = x_secant
        
        # Safety: ensure x_new is not NaN or Inf
        if not math.isfinite(x_new):
            x_new = x_secant
            if not math.isfinite(x_new):
                break
        
        f_new = safe_f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        # Update the three points: drop the one with largest |f|
        all_points = [(xa, fa), (xb, fb), (xc, fc), (x_new, f_new)]
        all_points.sort(key=lambda p: abs(p[1]))
        
        # Keep the three best with distinct x values
        kept = []
        for px, pf in all_points:
            if kept and abs(px - kept[-1][0]) < 1e-15 * max(1.0, abs(px)):
                if abs(pf) < abs(kept[-1][1]):
                    kept[-1] = (px, pf)
            else:
                kept.append((px, pf))
        
        if len(kept) >= 3:
            x, f0 = kept[0]
            x1, f1 = kept[1]
            x2, f2 = kept[2]
        else:
            break
    
    return best_x
# EVOLVE-BLOCK-END
