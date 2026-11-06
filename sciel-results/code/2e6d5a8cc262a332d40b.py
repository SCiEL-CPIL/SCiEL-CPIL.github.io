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
    
    # Phase 2: Hybrid Newton-Secant
    # We'll use a combination of Newton and secant methods
    # with damping for ill-conditioned problems
    
    x_prev = x
    x_curr = x + 1e-6 * max(1.0, abs(x))
    
    try:
        fx_prev = call_f(x_prev)
        fx_curr = call_f(x_curr)
        update_best(x_prev, fx_prev)
        update_best(x_curr, fx_curr)
    except:
        fx_prev = fx0
        fx_curr = None
    
    if fx_curr is None:
        # Try different perturbation
        x_curr = x - 1e-6 * max(1.0, abs(x))
        try:
            fx_curr = call_f(x_curr)
            update_best(x_curr, fx_curr)
        except:
            fx_curr = None
    
    # Main iteration loop
    for iteration in range(200):
        if not check_time() or eval_count >= MAX_EVALS:
            break
        if abs(best_fx) <= 1e-8:
            return best_x
        
        # Try Newton step if we can estimate derivative
        newton_x = None
        secant_x = None
        
        # Secant step
        if fx_prev is not None and fx_curr is not None and abs(fx_curr - fx_prev) > 1e-30:
            secant_x = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        
        # Newton step with finite difference derivative
        h = 1e-7 * max(1.0, abs(x_curr))
        try:
            fx_plus = call_f(x_curr + h)
            update_best(x_curr + h, fx_plus)
            if abs(fx_plus - fx_curr) > 1e-30:
                dfdx = (fx_plus - fx_curr) / h
                if abs(dfdx) > 1e-30:
                    newton_x = x_curr - fx_curr / dfdx
        except:
            pass
        
        # Choose step: prefer Newton if it's reasonable, else secant
        chosen_x = None
        if newton_x is not None and math.isfinite(newton_x):
            # Damping: don't move too far
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(newton_x - x_curr) <= max_step:
                chosen_x = newton_x
            else:
                # Damp the step
                direction = 1 if newton_x > x_curr else -1
                chosen_x = x_curr + direction * max_step
        elif secant_x is not None and math.isfinite(secant_x):
            max_step = 10.0 * max(1.0, abs(x_curr))
            if abs(secant_x - x_curr) <= max_step:
                chosen_x = secant_x
            else:
                direction = 1 if secant_x > x_curr else -1
                chosen_x = x_curr + direction * max_step
        
        if chosen_x is None:
            # Fallback: bisection-like step toward origin or away
            if abs(fx_curr) > 1e-8:
                chosen_x = x_curr - 0.5 * fx_curr * 1e-3
            else:
                break
        
        if not math.isfinite(chosen_x):
            break
        
        try:
            fx_new = call_f(chosen_x)
            update_best(chosen_x, fx_new)
        except:
            break
        
        if abs(fx_new) <= 1e-8:
            return chosen_x
        
        # Update for next iteration
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = chosen_x
        fx_curr = fx_new
        
        # Check convergence
        if abs(x_curr - x_prev) < 1e-12 * max(1.0, abs(x_curr)):
            break
    
    return best_x
# EVOLVE-BLOCK-END
