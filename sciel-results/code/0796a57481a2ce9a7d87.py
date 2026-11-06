"""Root finding for ill-conditioned problems."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find root of f near x0 using a robust hybrid approach.
    Returns x with |f(x)| <= 1e-8 or best effort within 500 evals.
    """
    start_time = time.time()
    DEADLINE = 130.0  # safety margin before 150s kill
    
    eval_count = 0
    MAX_EVALS = 500
    
    def call_f(x):
        nonlocal eval_count
        if eval_count >= MAX_EVALS:
            raise Exception("Max evaluations exceeded")
        result = f(x)
        eval_count += 1
        return result
    
    def check_time():
        return (time.time() - start_time) < DEADLINE
    
    # Track best point found
    best_x = float(x0)
    best_fx = None
    try:
        best_fx = call_f(best_x)
    except:
        pass
    
    def update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
    
    # Initial evaluation
    try:
        fx0 = call_f(x0)
        update_best(x0, fx0)
    except:
        fx0 = None
    
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0
    
    x = float(x0)
    
    # Phase 1: Try to find a bracket using expansion
    a, b, fa, fb = None, None, None, None
    if fx0 is not None:
        a, fa = x, fx0
        # Try expanding to find a sign change
        step = max(1e-6, abs(x) * 1e-6)
        for i in range(20):
            if not check_time() or eval_count >= MAX_EVALS:
                break
            b = a + step * (1 if fa > 0 else -1) * 2**i
            try:
                fb = call_f(b)
                update_best(b, fb)
                if abs(fb) <= 1e-8:
                    return b
                if fa * fb < 0:
                    break
                a, fa = b, fb
            except:
                break
        if fa is not None and fb is not None and fa * fb < 0:
            # We have a bracket, use bisection
            for _ in range(100):
                if not check_time() or eval_count >= MAX_EVALS:
                    break
                mid = (a + b) / 2.0
                try:
                    fm = call_f(mid)
                    update_best(mid, fm)
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                        break
                except:
                    break
            # Return the better of a, b
            if abs(fa) <= abs(fb):
                return a
            else:
                return b
    
    # Phase 2: Damped Newton with Line Search
    x_curr = x
    fx_curr = fx0
    
    # Estimate derivative
    h = 1e-7 * max(1.0, abs(x_curr))
    try:
        fx_plus = call_f(x_curr + h)
        fx_minus = call_f(x_curr - h)
        if math.isfinite(fx_plus) and math.isfinite(fx_minus):
            dfdx = (fx_plus - fx_minus) / (2 * h)
        else:
            dfdx = (fx_plus - fx_curr) / h
    except:
        try:
            fx_plus = call_f(x_curr + h)
            dfdx = (fx_plus - fx_curr) / h
        except:
            dfdx = 1.0
    
    if not math.isfinite(dfdx) or abs(dfdx) < 1e-15:
        dfdx = 1.0
    
    x_prev = x_curr
    fx_prev = fx_curr
    
    for iteration in range(100):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if abs(fx_curr) <= 1e-8:
            return x_curr
        
        # Newton step direction
        newton_step = -fx_curr / dfdx
        
        # Line search (Backtracking)
        current_abs_fx = abs(fx_curr)
        lambda_ = 1.0
        success = False
        x_next = x_curr
        fx_next = fx_curr
        
        for _ls in range(20):
            x_try = x_curr + lambda_ * newton_step
            if not math.isfinite(x_try):
                break
            try:
                fx_try = call_f(x_try)
                update_best(x_try, fx_try)
                if abs(fx_try) <= 1e-8:
                    return x_try
                if abs(fx_try) < current_abs_fx:
                    x_next = x_try
                    fx_next = fx_try
                    success = True
                    break
            except:
                break
            lambda_ *= 0.5
            
        if not success:
            # Fallback: Secant step using last two points
            if abs(fx_curr - fx_prev) > 1e-30 and abs(x_curr - x_prev) > 1e-30:
                secant_step = -fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
                x_try = x_curr + secant_step
                if math.isfinite(x_try):
                    try:
                        fx_try = call_f(x_try)
                        update_best(x_try, fx_try)
                        if abs(fx_try) <= 1e-8:
                            return x_try
                        if abs(fx_try) < current_abs_fx:
                            x_next = x_try
                            fx_next = fx_try
                            success = True
                    except:
                        pass
            
            if not success:
                break
        
        # Update state
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next
        
        # Update derivative estimate every few steps
        if iteration % 5 == 0:
            h = 1e-7 * max(1.0, abs(x_curr))
            try:
                fx_plus = call_f(x_curr + h)
                fx_minus = call_f(x_curr - h)
                if math.isfinite(fx_plus) and math.isfinite(fx_minus):
                    dfdx = (fx_plus - fx_minus) / (2 * h)
                else:
                    dfdx = (fx_plus - fx_curr) / h
            except:
                pass

    return best_x
# EVOLVE-BLOCK-END
