"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root x such that |f(x)| <= 1e-8 for oscillatory functions.
    
    Strategy:
    1. Try Newton's method first (fast if it works).
    2. If Newton fails, use multi-scale bracketing to find a sign change.
    3. Use Illinois method (modified bisection) for guaranteed convergence.
    
    All function calls are counted, so we minimize them.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    x0 = float(x0)
    
    # Helper to check time
    def time_ok():
        return (time.time() - start_time) < deadline
    
    best_x = x0
    best_f = None
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        if fx is not None and math.isfinite(fx):
            if best_f is None or abs(fx) < abs(best_f):
                best_f = fx
                best_x = x
    
    # Phase 1: Try Newton's method with limited iterations
    x = x0
    for i in range(15):
        if not time_ok():
            break
        
        try:
            fx = f(x)
        except Exception:
            break
        
        if fx is None or not math.isfinite(fx):
            break
        
        update_best(x, fx)
        
        if abs(fx) <= 1e-8:
            return x
        
        # Compute derivative using forward difference
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
        
        if fx_plus is None or not math.isfinite(fx_plus):
            break
        
        d = (fx_plus - fx) / h
        if d == 0 or not math.isfinite(d):
            break
        
        x_new = x - fx / d
        
        # Damping if step is too large
        max_step = 10.0 * max(1.0, abs(x))
        if abs(x_new - x) > max_step:
            x_new = x + math.copysign(max_step, x_new - x)
        
        x = x_new
    
    if not time_ok():
        return best_x
    
    # Phase 2: Multi-scale bracketing to find sign change
    brackets = []
    
    # First, evaluate f at x0
    try:
        fx0 = f(x0)
    except Exception:
        fx0 = None
    
    if fx0 is not None and math.isfinite(fx0):
        update_best(x0, fx0)
        
        if abs(fx0) <= 1e-8:
            return x0
        
        # Try expanding brackets in both directions
        for direction in [1.0, -1.0]:
            a = x0
            fa = fx0
            step = 0.1 * max(1.0, abs(x0))
            
            for i in range(25):
                if not time_ok():
                    break
                
                b = a + direction * step * (2.0 ** i)
                try:
                    fb = f(b)
                except Exception:
                    continue
                
                if fb is None or not math.isfinite(fb):
                    continue
                
                update_best(b, fb)
                
                if abs(fb) <= 1e-8:
                    return b
                
                if fa * fb < 0:
                    brackets.append((a, b, fa, fb))
                    break
                
                if abs(b) > 1e6:
                    break
    
    # If no bracket found, try a grid search
    if not brackets and time_ok():
        scale = max(1.0, abs(x0))
        test_points = []
        
        for i in range(1, 8):
            test_points.append(x0 + i * scale)
            test_points.append(x0 - i * scale)
        
        for i in range(1, 4):
            test_points.append(x0 + i * 0.1 * scale)
            test_points.append(x0 - i * 0.1 * scale)
        
        test_points = sorted(set(test_points))
        
        prev_x = None
        prev_f = None
        
        for px in test_points:
            if not time_ok():
                break
            
            try:
                pf = f(px)
            except Exception:
                continue
            
            if pf is None or not math.isfinite(pf):
                continue
            
            update_best(px, pf)
            
            if abs(pf) <= 1e-8:
                return px
            
            if prev_x is not None and prev_f is not None and prev_f * pf < 0:
                brackets.append((prev_x, px, prev_f, pf))
            
            prev_x = px
            prev_f = pf
    
    # Phase 3: Use Illinois method on the best bracket
    if brackets and time_ok():
        brackets.sort(key=lambda b: abs(b[1] - b[0]))
        
        for a, b, fa, fb in brackets:
            if not time_ok():
                break
            
            # Illinois method
            for i in range(100):
                if not time_ok():
                    break
                
                if fa * fb > 0:
                    break
                
                c = (a + b) / 2.0
                try:
                    fc = f(c)
                except Exception:
                    break
                
                if fc is None or not math.isfinite(fc):
                    break
                
                update_best(c, fc)
                
                if abs(fc) <= 1e-8:
                    return c
                
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
                
                # Illinois modification
                if abs(fa) > abs(fb):
                    a = (a + b) / 2.0
                    try:
                        fa = f(a)
                    except Exception:
                        break
                    if fa is None or not math.isfinite(fa):
                        break
                    update_best(a, fa)
                else:
                    b = (a + b) / 2.0
                    try:
                        fb = f(b)
                    except Exception:
                        break
                    if fb is None or not math.isfinite(fb):
                        break
                    update_best(b, fb)
            
            if best_f is not None and abs(best_f) <= 1e-8:
                return best_x
    
    return best_x
# EVOLVE-BLOCK-END
