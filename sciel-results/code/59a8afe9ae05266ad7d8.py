"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    # Deadline guard: 120s wall-clock limit, use 110s as safety margin
    start_time = time.time()
    DEADLINE = 110.0
    MAX_CALLS = 490  # Stay under 500
    
    call_count = [0]
    
    def feval(x):
        """Counted function evaluation with budget check."""
        if call_count[0] >= MAX_CALLS:
            return None
        if time.time() - start_time > DEADLINE:
            return None
        call_count[0] += 1
        return f(x)
    
    def is_good(x, fx):
        """Check if x is a valid root candidate."""
        if fx is None:
            return False
        if not math.isfinite(x) or not math.isfinite(fx):
            return False
        return abs(fx) <= 1e-8
    
    x = float(x0)
    
    # First evaluation at x0
    fx = feval(x)
    if fx is None:
        return x
    if is_good(x, fx):
        return x
    
    best_x = x
    best_fx = fx
    best_abs = abs(fx)
    
    def update_best(x_new, fx_new):
        nonlocal best_x, best_fx, best_abs
        if fx_new is not None and math.isfinite(x_new) and math.isfinite(fx_new):
            if abs(fx_new) < best_abs:
                best_abs = abs(fx_new)
                best_x = x_new
                best_fx = fx_new
        if is_good(x_new, fx_new):
            return x_new
        return None
    
    # Check deadline
    if time.time() - start_time > DEADLINE:
        return best_x
    
    # Phase 1: Try Newton's method with finite difference derivative
    # This is fast for smooth polynomials when starting near a root
    x_newton = x
    fx_newton = fx
    converged = False
    
    for i in range(30):
        if time.time() - start_time > DEADLINE:
            break
        if call_count[0] >= MAX_CALLS - 5:
            break
        
        h = 1e-7 * max(1.0, abs(x_newton))
        fx_plus = feval(x_newton + h)
        if fx_plus is None:
            break
        
        # Central difference for better accuracy
        fx_minus = feval(x_newton - h)
        if fx_minus is None:
            break
        
        if abs(h) < 1e-15:
            break
        
        # Use central difference
        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            # Fallback to forward difference
            d = (fx_plus - fx_newton) / h
            if d == 0 or not math.isfinite(d):
                break
        
        # Damped Newton step
        dx = -fx_newton / d
        # Limit step size to prevent divergence
        max_step = 10.0 * max(1.0, abs(x_newton))
        if abs(dx) > max_step:
            dx = math.copysign(max_step, dx)
        
        x_candidate = x_newton + dx
        fx_candidate = feval(x_candidate)
        if fx_candidate is None:
            break
        
        result = update_best(x_candidate, fx_candidate)
        if result is not None:
            return result
        
        if abs(fx_candidate) < 1e-12:
            return x_candidate
        
        # Check convergence
        if abs(dx) < 1e-14 * max(1.0, abs(x_candidate)):
            converged = True
            break
        
        x_newton = x_candidate
        fx_newton = fx_candidate
    
    if time.time() - start_time > DEADLINE or call_count[0] >= MAX_CALLS - 3:
        return best_x
    
    # Phase 2: If Newton didn't converge, try bisection with bracketing
    # First, find a bracket [a, b] where f(a) and f(b) have opposite signs
    a = x - 1.0
    b = x + 1.0
    fa = None
    fb = None
    
    # Try to find a bracket by expanding
    step = 1.0
    found_bracket = False
    
    for i in range(20):
        if time.time() - start_time > DEADLINE:
            break
        if call_count[0] >= MAX_CALLS - 10:
            break
        
        a = x - step
        b = x + step
        fa = feval(a)
        if fa is None:
            break
        fb = feval(b)
        if fb is None:
            break
        
        update_best(a, fa)
        update_best(b, fb)
        
        if fa is not None and fb is not None:
            if math.isfinite(fa) and math.isfinite(fb):
                if fa * fb < 0:
                    found_bracket = True
                    break
        
        step *= 2.0
    
    if time.time() - start_time > DEADLINE or call_count[0] >= MAX_CALLS - 5:
        return best_x
    
    if found_bracket:
        # Bisection method
        for i in range(50):
            if time.time() - start_time > DEADLINE:
                break
            if call_count[0] >= MAX_CALLS - 2:
                break
            
            mid = 0.5 * (a + b)
            fm = feval(mid)
            if fm is None:
                break
            
            result = update_best(mid, fm)
            if result is not None:
                return result
            
            if abs(fm) < 1e-12:
                return mid
            
            if fa * fm < 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
            
            if abs(b - a) < 1e-15:
                break
    
    if time.time() - start_time > DEADLINE or call_count[0] >= MAX_CALLS - 3:
        return best_x
    
    # Phase 3: If still not found, try a few more Newton iterations from best point
    x_try = best_x
    fx_try = best_fx
    
    for i in range(15):
        if time.time() - start_time > DEADLINE:
            break
        if call_count[0] >= MAX_CALLS - 2:
            break
        
        h = 1e-7 * max(1.0, abs(x_try))
        fx_plus = feval(x_try + h)
        if fx_plus is None:
            break
        
        d = (fx_plus - fx_try) / h
        if d == 0 or not math.isfinite(d):
            break
        
        dx = -fx_try / d
        max_step = 5.0 * max(1.0, abs(x_try))
        if abs(dx) > max_step:
            dx = math.copysign(max_step, dx)
        
        x_candidate = x_try + dx
        fx_candidate = feval(x_candidate)
        if fx_candidate is None:
            break
        
        result = update_best(x_candidate, fx_candidate)
        if result is not None:
            return result
        
        if abs(fx_candidate) < 1e-12:
            return x_candidate
        
        if abs(dx) < 1e-14 * max(1.0, abs(x_candidate)):
            break
        
        x_try = x_candidate
        fx_try = fx_candidate
    
    return best_x
# EVOLVE-BLOCK-END
