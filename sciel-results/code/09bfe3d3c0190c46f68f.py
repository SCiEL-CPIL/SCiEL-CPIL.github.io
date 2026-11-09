"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 10.0  # Safety margin for 120s limit (3 equations, ~10s each)
    
    def time_ok():
        return (time.time() - start_time) < deadline
    
    x = float(x0)
    if not math.isfinite(x):
        x = 0.0
    
    # Evaluate initial point
    fx = f(x)
    if not math.isfinite(fx):
        # Try some other points
        for dx in [1.0, -1.0, 0.1, -0.1, 0.01, -0.01]:
            x_try = x + dx
            try:
                fx_try = f(x_try)
                if math.isfinite(fx_try):
                    x, fx = x_try, fx_try
                    break
            except:
                pass
    
    if abs(fx) <= 1e-8:
        return x
    
    # Phase 1: Bracket finding using a denser local scan
    # For oscillatory functions, roots are frequent, so a local scan works well
    a = x
    fa = fx
    
    # Try to find a bracket by scanning outward
    # Start with small steps and increase
    step_sizes = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    for h in step_sizes:
        if not time_ok():
            break
        
        # Try positive direction
        b = x + h
        try:
            fb = f(b)
            if math.isfinite(fb):
                if fa * fb < 0:
                    # Found a bracket
                    break
                # If |fb| is smaller, move a to b
                if abs(fb) < abs(fa):
                    a, fa = b, fb
        except:
            pass
        
        # Try negative direction
        b = x - h
        try:
            fb = f(b)
            if math.isfinite(fb):
                if fa * fb < 0:
                    # Found a bracket
                    break
                # If |fb| is smaller, move a to b
                if abs(fb) < abs(fa):
                    a, fa = b, fb
        except:
            pass
    
    # Phase 2: Refine using Illinois algorithm (derivative-free, robust)
    if 'a' in locals() and 'fa' in locals() and 'b' in locals() and 'fb' in locals():
        if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
            # Illinois algorithm
            for _ in range(100):
                if not time_ok():
                    break
                
                # Check if we've converged
                if abs(fa) <= 1e-10:
                    return a
                if abs(fb) <= 1e-10:
                    return b
                
                # Illinois: weighted bisection
                # If fa and fb have same sign after update, shrink the larger interval
                m = (a + b) / 2.0
                try:
                    fm = f(m)
                except:
                    break
                
                if not math.isfinite(fm):
                    break
                
                if abs(fm) <= 1e-10:
                    return m
                
                if fa * fm < 0:
                    b, fb = m, fm
                else:
                    a, fa = m, fm
                
                # Illinois: if the function values at the endpoints have the same sign
                # as the midpoint, shrink the interval on the side with the larger |f|
                if abs(fa) > abs(fb):
                    # Shrink a-side
                    a = a + (b - a) * 0.5
                else:
                    # Shrink b-side
                    b = b - (b - a) * 0.5
                
                # Ensure we don't cross
                if a > b:
                    a, b = b, a
                    fa, fb = fb, fa
        else:
            # No valid bracket, try Newton's method as fallback
            x = float(x0)
            for _ in range(60):
                if not time_ok():
                    break
                fx = f(x)
                if abs(fx) <= 1e-8:
                    return x
                h = 1e-6 * max(1.0, abs(x))
                try:
                    d = (f(x + h) - fx) / h
                except:
                    break
                if d == 0 or not math.isfinite(d):
                    break
                x = x - fx / d
                if not math.isfinite(x):
                    break
            return x
    else:
        # No bracket found, try Newton's method as fallback
        x = float(x0)
        for _ in range(60):
            if not time_ok():
                break
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            try:
                d = (f(x + h) - fx) / h
            except:
                break
            if d == 0 or not math.isfinite(d):
                break
            x = x - fx / d
            if not math.isfinite(x):
                break
        return x
# EVOLVE-BLOCK-END
