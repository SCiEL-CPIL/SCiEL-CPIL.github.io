"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    start_time = time.time()
    DEADLINE = start_time + 110.0
    
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
            if abs(val) < best_abs:
                best_abs = abs(val)
                best_x = x
            return val
        except Exception:
            return float('inf')
    
    x = float(x0)
    fx = safe_f(x)
    
    if abs(fx) <= TOL:
        return x
    
    # Try to find a bracket using adaptive steps
    # Start with small steps and expand
    h = 1e-3
    bracket_a = None
    bracket_b = None
    fa = None
    fb = None
    
    # Check nearby points for a sign change
    for i in range(50):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        x_right = x + h
        x_left = x - h
        fr = safe_f(x_right)
        fl = safe_f(x_left)
        
        if abs(fr) <= TOL:
            return x_right
        if abs(fl) <= TOL:
            return x_left
        
        if math.isfinite(fx) and math.isfinite(fr):
            if fx * fr < 0:
                bracket_a = x
                bracket_b = x_right
                fa = fx
                fb = fr
                break
        if math.isfinite(fx) and math.isfinite(fl):
            if fx * fl < 0:
                bracket_a = x_left
                bracket_b = x
                fa = fl
                fb = fx
                break
        
        h *= 2.0
        if h > 1e6:
            break
    
    # If we found a bracket, use bisection
    if bracket_a is not None and bracket_b is not None:
        a = bracket_a
        b = bracket_b
        fa = safe_f(a)
        fb = safe_f(b)
        
        if abs(fa) <= TOL:
            return a
        if abs(fb) <= TOL:
            return b
        
        # Bisection
        for _ in range(60):
            if calls >= MAX_CALLS or time.time() > DEADLINE:
                break
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if abs(fm) <= TOL:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        # Return the best of a, b, mid
        candidates = [a, b, (a+b)/2.0]
        best_val = float('inf')
        best_pt = a
        for c in candidates:
            if math.isfinite(c):
                fc = safe_f(c)
                if abs(fc) < best_val:
                    best_val = abs(fc)
                    best_pt = c
        return best_pt
    
    # If no bracket found, try secant method with fallback
    # Use two nearby points
    h = 1e-4
    x1 = x
    x2 = x + h
    f1 = safe_f(x1)
    f2 = safe_f(x2)
    
    if abs(f1) <= TOL:
        return x1
    if abs(f2) <= TOL:
        return x2
    
    # Secant iterations
    for _ in range(40):
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            break
        if f2 - f1 == 0:
            break
        x3 = x2 - f2 * (x2 - x1) / (f2 - f1)
        f3 = safe_f(x3)
        if abs(f3) <= TOL:
            return x3
        x1 = x2
        f1 = f2
        x2 = x3
        f2 = f3
    
    # Return best found
    return best_x
# EVOLVE-BLOCK-END
