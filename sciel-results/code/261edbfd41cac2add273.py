import math
import time
import sys
from scipy.optimize import brentq

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    DEADLINE = time.time() + 140
    
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
            if abs_val <= TOL:
                return val
            return val
        except Exception:
            return float('inf')

    # Initial evaluation
    fx0 = safe_f(x0)
    if abs(fx0) <= TOL:
        return x0
    
    # Spectral-aware bracketing: estimate dominant period, then bracket efficiently
    bracket_found = False
    a = x0
    b = None
    fa = fx0
    
    # Phase 1: Quick local scan to catch nearby roots (cheap, 8 calls max)
    small_offsets = [0.01, -0.01, 0.05, -0.05, 0.1, -0.1, 0.5, -0.5]
    for d in small_offsets:
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x_new = x0 + d
        f_new = safe_f(x_new)
        if math.isinf(f_new):
            continue
        if fa * f_new < 0:
            a = x0
            b = x_new
            bracket_found = True
            break
    
    # Phase 2: Estimate dominant period via adaptive frequency sweep
    if not bracket_found:
        # Use a finer grid of candidate frequencies for better period estimation
        candidate_k = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        
        best_period = None
        best_oscillation = 0.0
        
        for k in candidate_k:
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            period = 2.0 * math.pi / k
            
            # Evaluate at quarter-period offsets to detect oscillation
            x_plus = x0 + period / 4.0
            x_minus = x0 - period / 4.0
            fp = safe_f(x_plus)
            fm = safe_f(x_minus)
            
            if math.isinf(fp) or math.isinf(fm):
                continue
            
            # Measure oscillation: difference from f(x0)
            osc = abs(fp - fa) + abs(fm - fa)
            if osc > best_oscillation:
                best_oscillation = osc
                best_period = period
        
        # Phase 3: Use estimated period to find brackets with adaptive refinement
        if best_period is not None:
            half_period = best_period / 2.0
            quarter_period = best_period / 4.0
            
            # First pass: try half-period offsets
            for n in range(1, 6):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                for sign in [1, -1]:
                    x_new = x0 + sign * n * half_period
                    f_new = safe_f(x_new)
                    if math.isinf(f_new):
                        continue
                    if fa * f_new < 0:
                        a = x0
                        b = x_new
                        bracket_found = True
                        break
                if bracket_found:
                    break
            
            # Second pass: try quarter-period offsets if half-period failed
            if not bracket_found:
                for n in range(1, 10):
                    if calls >= MAX_CALLS or time.time() > DEADLINE:
                        break
                    for sign in [1, -1]:
                        x_new = x0 + sign * n * quarter_period
                        f_new = safe_f(x_new)
                        if math.isinf(f_new):
                            continue
                        if fa * f_new < 0:
                            a = x0
                            b = x_new
                            bracket_found = True
                            break
                    if bracket_found:
                        break
            
            # Third pass: try eighth-period offsets for very fine structure
            if not bracket_found:
                eighth_period = best_period / 8.0
                for n in range(1, 12):
                    if calls >= MAX_CALLS or time.time() > DEADLINE:
                        break
                    for sign in [1, -1]:
                        x_new = x0 + sign * n * eighth_period
                        f_new = safe_f(x_new)
                        if math.isinf(f_new):
                            continue
                        if fa * f_new < 0:
                            a = x0
                            b = x_new
                            bracket_found = True
                            break
                    if bracket_found:
                        break
        
        # Phase 4: Fallback to geometric expansion if spectral method didn't find bracket
        if not bracket_found:
            for d in [1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0, 50.0, -50.0]:
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                x_new = x0 + d
                f_new = safe_f(x_new)
                if math.isinf(f_new):
                    continue
                if fa * f_new < 0:
                    a = x0
                    b = x_new
                    bracket_found = True
                    break
    
    if bracket_found:
        try:
            root = brentq(safe_f, a, b, xtol=1e-12, rtol=1e-12, maxiter=30)
            if math.isfinite(root):
                return root
        except Exception:
            pass
        
        # Fallback to bisection if brentq fails
        try:
            lo, hi = min(a, b), max(a, b)
            flo, fhi = safe_f(lo), safe_f(hi)
            if flo * fhi > 0:
                raise ValueError("No sign change")
            for _ in range(40):
                if calls >= MAX_CALLS or time.time() > DEADLINE:
                    break
                mid = (lo + hi) / 2
                fmid = safe_f(mid)
                if abs(fmid) <= TOL:
                    return mid
                if flo * fmid < 0:
                    hi = mid
                    fhi = fmid
                else:
                    lo = mid
                    flo = fmid
            return (lo + hi) / 2
        except Exception:
            pass
    
    # If no bracket found, try Newton's method from the best point found
    x = best_x
    prev_x = None
    for _ in range(30):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
        
        h = 1e-7 * max(1.0, abs(x))
        f_minus = safe_f(x - h)
        f_plus = safe_f(x + h)
        if math.isinf(f_minus) or math.isinf(f_plus):
            break
        d = (f_plus - f_minus) / (2 * h)
        if abs(d) < 1e-15 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
            
        # Damping if step is too large
        max_step = 5.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
             direction = 1 if x_new > x else -1
             x_new = x + direction * max_step
             
        # If we're oscillating, break
        if prev_x is not None and abs(x_new - x) < 1e-10 * max(1.0, abs(x)):
            break
        prev_x = x
        x = x_new
        
    return best_x
