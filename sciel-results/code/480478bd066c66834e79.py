import math
import time

def find_root(f, x0):
    """
    Finds a root of f starting near x0.
    Strategy:
    1. Try Newton's method steps, tracking history.
    2. If a sign change is detected in the last two points, switch to Brent's method 
       on that bracket for guaranteed and fast convergence.
    3. If Newton steps are not making progress or fail, expand a bracket around x0 
       to find a sign change, then use Brent's method.
    """
    x0 = float(x0)
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin for 120s limit? 
    # Actually, the limit is 150s kill, 120s finish. 
    # Let's use 100s to be safe.
    
    # State
    x = x0
    fx = f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # History for sign change detection
    # (x_prev, f_prev)
    x_prev = None
    f_prev = None
    
    # Newton iteration count
    newton_steps = 0
    max_newton_steps = 10
    
    h = 1e-4 * max(1.0, abs(x0))
    
    # Phase 1: Newton with sign-change check
    for i in range(max_newton_steps):
        if time.time() > deadline:
            break
            
        # Check convergence
        if abs(fx) <= 1e-8:
            return x
            
        # Estimate derivative
        # Use central difference if possible, or forward
        h_step = 1e-6 * max(1.0, abs(x))
        # We need f(x+h) and f(x-h) ideally, but that's 2 calls.
        # We can reuse fx.
        # Let's use forward difference to save calls? 
        # Central is better for accuracy.
        # Let's try to be economical.
        
        # If we have x_prev, we can use secant derivative?
        # (fx - f_prev) / (x - x_prev)
        
        if x_prev is not None and x != x_prev:
            d = (fx - f_prev) / (x - x_prev)
        else:
            # Finite difference
            x_h = x + h_step
            fx_h = f(x_h)
            d = (fx_h - fx) / h_step
            # We could also do central, but let's stick to 1 extra call for deriv
            # if we don't have prev.
            
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
            
        # Safety: limit step size
        if abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            break
            
        # Move
        x_prev, f_prev = x, fx
        x = x_new
        fx = f(x)
        newton_steps += 1
        
        # Check for sign change between x_prev and x
        if f_prev * fx < 0:
            # We have a bracket [min(x_prev, x), max(x_prev, x)]
            a, b = sorted([x_prev, x])
            # Use Brent's method
            try:
                from scipy.optimize import brentq
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
                return root
            except Exception:
                # If brentq fails, fallback to bisection manually?
                # Or just return the better of the two
                if abs(fx) < abs(f_prev):
                    return x
                else:
                    return x_prev

    # Phase 2: If Newton didn't converge or find bracket, find bracket by expansion
    # We are at x, fx.
    # Let's expand around x0 or current x? 
    # Expanding around current x is safer if we moved.
    # But if we moved far and wrong, maybe expand around x0?
    # Let's expand around current x.
    
    # We need a bracket.
    # Current point: x, fx
    # Try x + h, x - h
    
    a, b = None, None
    fa, fb = None, None
    
    # We already have x, fx.
    # Let's try to find a sign change with x.
    
    # Try expanding in both directions
    step = h
    max_expansions = 20
    
    for i in range(max_expansions):
        if time.time() > deadline:
            break
            
        # Try right
        x_r = x + step
        fx_r = f(x_r)
        
        if fx * fx_r < 0:
            a, b = sorted([x, x_r])
            fa, fb = fx, fx_r
            break
            
        # Try left
        x_l = x - step
        fx_l = f(x_l)
        
        if fx * fx_l < 0:
            a, b = sorted([x, x_l])
            fa, fb = fx, fx_l
            break
            
        # No sign change, expand
        step *= 2.0
        
        # Safety: if step becomes huge, break
        if step > 1e10:
            break

    if a is not None:
        try:
            from scipy.optimize import brentq
            root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=50)
            return root
        except Exception:
            pass
            
    # Fallback: return best guess
    if abs(fx) <= 1e-8:
        return x
        
    # If we have a previous point that was closer
    if f_prev is not None and abs(f_prev) < abs(fx):
        return x_prev
        
    return x
