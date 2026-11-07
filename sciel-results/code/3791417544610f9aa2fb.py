"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    TOL = 1e-8
    MAX_CALLS = 480
    DEADLINE = time.time() + 110.0
    
    calls = 0
    best_x = float(x0)
    best_abs = float('inf')
    
    def safe_f(x):
        nonlocal calls, best_x, best_abs
        if calls >= MAX_CALLS or time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
        except Exception:
            return float('inf')
        if not math.isfinite(val):
            return float('inf')
        if abs(val) < best_abs:
            best_abs = abs(val)
            best_x = x
        return val
    
    x = float(x0)
    fx = safe_f(x)
    
    if abs(fx) <= TOL:
        return x
    
    # Try Newton-Raphson with finite difference derivative
    for _ in range(30):
        if time.time() > DEADLINE or calls >= MAX_CALLS:
            break
        if not math.isfinite(fx):
            break
        h = 1e-6 * max(1.0, abs(x))
        fx_plus = safe_f(x + h)
        if not math.isfinite(fx_plus):
            break
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
        fx = safe_f(x)
        if abs(fx) <= TOL:
            return x
    
    if abs(fx) <= TOL:
        return x
    
    # Try to bracket a root using adaptive expansion
    # First, try small steps in both directions from x0
    found_bracket = False
    step = 1e-3
    for direction in [1, -1]:
        if time.time() > DEADLINE or calls >= MAX_CALLS:
            break
        a = float(x0)
        fa = safe_f(a)
        b = x0 + direction * step
        fb = safe_f(b)
        
        # Expand until we find a sign change or give up
        for _ in range(200):
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
            if not math.isfinite(fa) or not math.isfinite(fb):
                break
            if fa * fb < 0:
                found_bracket = True
                break
            # Move the endpoint that has smaller magnitude
            if abs(fa) < abs(fb):
                a = b
                fa = fb
                b = b + direction * step
                fb = safe_f(b)
            else:
                b = a
                fb = fa
                a = a + direction * step
                fa = safe_f(a)
            step *= 2.0
            if step > 1e6:
                break
        
        if found_bracket:
            break
    
    if found_bracket and time.time() < DEADLINE and calls < MAX_CALLS:
        # Use bisection to converge
        for _ in range(50):
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
            mid = (a + b) / 2.0
            fm = safe_f(mid)
            if not math.isfinite(fm):
                break
            if abs(fm) <= TOL:
                return mid
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            if abs(b - a) < 1e-12:
                break
        
        # Return the best of the bracket endpoints or midpoint
        candidates = [a, b, (a + b) / 2.0]
        best_c = min(candidates, key=lambda c: abs(safe_f(c)) if time.time() < DEADLINE and calls < MAX_CALLS else float('inf'))
        return best_c
    
    # If no bracket found, try a few more strategies
    # Try secant method from x0
    if time.time() < DEADLINE and calls < MAX_CALLS:
        x1 = x0 + 1e-3
        f1 = safe_f(x1)
        x0_sec = x0
        f0_sec = fx
        
        for _ in range(40):
            if time.time() > DEADLINE or calls >= MAX_CALLS:
                break
            if not math.isfinite(f0_sec) or not math.isfinite(f1):
                break
            if abs(f1 - f0_sec) < 1e-15:
                break
            x_new = x1 - f1 * (x1 - x0_sec) / (f1 - f0_sec)
            if not math.isfinite(x_new):
                break
            x0_sec, f0_sec = x1, f1
            x1 = x_new
            f1 = safe_f(x1)
            if abs(f1) <= TOL:
                return x1
    
    # Return the best x found
    return best_x
# EVOLVE-BLOCK-END
