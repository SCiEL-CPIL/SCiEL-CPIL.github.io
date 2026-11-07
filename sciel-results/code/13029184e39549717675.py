import math
import time
from scipy.optimize import brentq
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0

    x = float(x0)
    try:
        fx = f(x)
    except:
        return x

    if not math.isfinite(fx):
        return x

    best_x = x
    best_f = fx
    best_abs = abs(fx)

    if best_abs <= 1e-8:
        return x

    # --- Adaptive Frequency Sweep to estimate dominant period ---
    # Sample f at points spaced by 1/(2*pi*k) for several k values
    # to identify which frequency gives the most sign changes (i.e., dominant period)
    
    # Try a range of angular frequencies k
    k_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]
    
    best_period = None
    best_sign_changes = 0
    
    for k in k_values:
        if time.time() > deadline:
            break
        
        # Sampling interval: delta = 1/k (since period = 2*pi/k, we sample at 1/k which is ~1/(2*pi)*2*pi/k)
        # Actually, to detect period T, we need step < T/2. If T = 2*pi/k, step should be < pi/k.
        # Let's use step = pi/k to get about 2 samples per period.
        step = math.pi / k
        
        # Sample in a window around x0
        # Window size: a few periods
        window = 4 * step  # 4 periods
        
        # Generate sample points
        n_samples = 10
        points = []
        for i in range(n_samples):
            # Spread points symmetrically around x0
            offset = (i - n_samples // 2) * step
            px = x + offset
            points.append(px)
        
        # Evaluate f at these points
        vals = []
        valid = True
        for px in points:
            try:
                pv = f(px)
                if not math.isfinite(pv):
                    valid = False
                    break
                vals.append(pv)
                if abs(pv) < best_abs:
                    best_abs = abs(pv)
                    best_x = px
                    best_f = pv
            except:
                valid = False
                break
        
        if not valid:
            continue
        
        # Count sign changes
        sign_changes = 0
        for i in range(len(vals) - 1):
            if vals[i] * vals[i+1] < 0:
                sign_changes += 1
        
        if sign_changes > best_sign_changes:
            best_sign_changes = sign_changes
            best_period = 2 * math.pi / k
    
    # If no clear period found, fall back to derivative-based estimate
    if best_period is None or best_period <= 0:
        h_est = 1e-5 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h_est)
            fx_minus = f(x - h_est)
            if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                slope = (fx_plus - fx_minus) / (2 * h_est)
                abs_slope = abs(slope)
            else:
                abs_slope = 1.0
        except:
            abs_slope = 1.0
        
        if abs_slope > 1e-3:
            best_period = 2 * math.pi / abs_slope
        else:
            best_period = 1.0
    
    # Use estimated period to place dense brackets
    # We want step size = period / 4 to ensure we catch sign changes
    step_size = best_period / 4.0
    
    # Ensure step_size is reasonable
    step_size = max(1e-6, min(step_size, 100.0))
    
    # Generate search points around x0
    # Use a symmetric search with increasing radius
    search_radius = 10 * best_period  # Search 10 periods in each direction
    
    # Create a list of candidate x values
    candidates = [x]
    
    # Add points on a grid with step_size, but only within search_radius
    # Use both positive and negative directions
    n_steps = int(search_radius / step_size) + 1
    
    # Build sorted list of unique x values
    x_vals = set()
    for i in range(-n_steps, n_steps + 1):
        xv = x + i * step_size
        x_vals.add(xv)
    
    x_vals = sorted(x_vals)
    
    # Evaluate f at each candidate and look for sign changes
    bracket_found = False
    bracket_lo, bracket_hi = 0, 0
    bracket_flo, bracket_fhi = 0, 0
    
    prev_x = None
    prev_f = None
    
    for xv in x_vals:
        if time.time() > deadline:
            break
        
        try:
            fv = f(xv)
        except:
            prev_x = None
            prev_f = None
            continue
        
        if not math.isfinite(fv):
            prev_x = None
            prev_f = None
            continue
        
        abs_fv = abs(fv)
        if abs_fv < best_abs:
            best_abs = abs_fv
            best_x = xv
            best_f = fv
        
        if abs_fv <= 1e-8:
            return xv
        
        if prev_x is not None and prev_f is not None:
            if prev_f * fv < 0:
                bracket_found = True
                bracket_lo = prev_x
                bracket_hi = xv
                bracket_flo = prev_f
                bracket_fhi = fv
                break
        
        prev_x = xv
        prev_f = fv
    
    # If no bracket found with dense grid, fall back to geometric expansion
    if not bracket_found:
        # Geometric expansion as before
        initial_step = step_size
        deltas = []
        d = initial_step
        while d <= 1e6:
            deltas.append(d)
            d *= 2.0
        
        small_deltas = [1e-6, 1e-5, 1e-4]
        for sd in small_deltas:
            if sd < initial_step:
                deltas.insert(0, sd)
        
        deltas = sorted(set(deltas))
        
        prev_x_right = x
        prev_f_right = fx
        prev_x_left = x
        prev_f_left = fx
        
        for d in deltas:
            if time.time() > deadline:
                break
            
            # Right side
            x_r = x + d
            try:
                f_r = f(x_r)
            except:
                f_r = None
            
            if f_r is not None and math.isfinite(f_r):
                abs_f_r = abs(f_r)
                if abs_f_r < best_abs:
                    best_abs = abs_f_r
                    best_x = x_r
                    best_f = f_r
                
                if abs_f_r <= 1e-8:
                    return x_r
                
                if prev_f_right is not None and math.isfinite(prev_f_right):
                    if prev_f_right * f_r < 0:
                        bracket_found = True
                        bracket_lo = prev_x_right
                        bracket_hi = x_r
                        bracket_flo = prev_f_right
                        bracket_fhi = f_r
                        break
                
                prev_x_right = x_r
                prev_f_right = f_r
            
            # Left side
            x_l = x - d
            try:
                f_l = f(x_l)
            except:
                f_l = None
            
            if f_l is not None and math.isfinite(f_l):
                abs_f_l = abs(f_l)
                if abs_f_l < best_abs:
                    best_abs = abs_f_l
                    best_x = x_l
                    best_f = f_l
                
                if abs_f_l <= 1e-8:
                    return x_l
                
                if prev_f_left is not None and math.isfinite(prev_f_left):
                    if prev_f_left * f_l < 0:
                        bracket_found = True
                        bracket_lo = x_l
                        bracket_hi = prev_x_left
                        bracket_flo = f_l
                        bracket_fhi = prev_f_left
                        break
                
                prev_x_left = x_l
                prev_f_left = f_l
    
    if bracket_found:
        try:
            root = brentq(f, bracket_lo, bracket_hi, xtol=1e-12, rtol=1e-12, maxiter=50)
            if time.time() < deadline:
                f_root = f(root)
                if math.isfinite(f_root) and abs(f_root) <= 1e-8:
                    return root
        except Exception:
            pass
        
        # Fallback bisection
        a_b, b_b = min(bracket_lo, bracket_hi), max(bracket_lo, bracket_hi)
        fa_b, fb_b = bracket_flo, bracket_fhi
        for _ in range(50):
            if time.time() > deadline:
                break
            m = (a_b + b_b) / 2.0
            try:
                fm = f(m)
            except:
                break
            if not math.isfinite(fm):
                break
            if abs(fm) <= 1e-8:
                return m
            if fa_b * fm < 0:
                b_b = m
                fb_b = fm
            else:
                a_b = m
                fa_b = fm
        return (a_b + b_b) / 2.0

    # If no bracket, refine best_x with secant method
    if best_abs > 1e-8 and time.time() < deadline - 5:
        h = 1e-5 * max(1.0, abs(best_x))
        x1 = best_x
        f1 = best_f
        x2 = best_x + h
        try:
            f2 = f(x2)
        except:
            f2 = None
        if f2 is not None and math.isfinite(f2):
            for _ in range(5):
                if time.time() > deadline:
                    break
                if abs(f2 - f1) < 1e-15:
                    break
                x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
                try:
                    f3 = f(x3)
                except:
                    break
                if not math.isfinite(f3):
                    break
                if abs(f3) < best_abs:
                    best_abs = abs(f3)
                    best_x = x3
                    best_f = f3
                    if best_abs <= 1e-8:
                        return x3
                x1, f1 = x2, f2
                x2, f2 = x3, f3

    return best_x
# EVOLVE-BLOCK-END
