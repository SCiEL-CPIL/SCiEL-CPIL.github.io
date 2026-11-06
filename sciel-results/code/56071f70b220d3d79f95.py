"""Root finding for ill-conditioned functions using a hybrid approach."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Returns x with |f(x)| <= 1e-8 using at most 500 evaluations.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    max_evals = 490  # Safety margin for 500 limit
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        if eval_count >= max_evals:
            return float('nan')
        eval_count += 1
        return f(x)
    
    x = float(x0)
    fx = eval_f(x)
    
    if abs(fx) <= 1e-8:
        return x
    
    # Try multiple strategies in sequence
    
    # Strategy 1: Secant method (fast when it works)
    x1 = x
    fx1 = fx
    # Perturb for second point
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = eval_f(x2)
    
    if abs(fx2) <= 1e-8:
        return x2
    
    # Secant iterations
    best_x = x if abs(fx) < abs(fx2) else x2
    best_fx = fx if abs(fx) < abs(fx2) else fx2
    
    for i in range(40):
        if time.time() > deadline:
            break
        if eval_count >= max_evals:
            break
        
        if fx1 == fx2:
            break
        
        # Secant step
        x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
        
        # Guard against divergence
        if abs(x_new - x2) > 10 * max(1.0, abs(x2)):
            x_new = (x1 + x2) / 2.0
        
        fx_new = eval_f(x_new)
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
        
        # Shift points
        x1, fx1 = x2, fx2
        x2, fx2 = x_new, fx_new
    
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # Strategy 2: Bisection with bracket expansion
    # Try to find a bracket around x0
    a = x
    fa = fx
    b = x
    fb = fx
    
    # Expand to find sign change
    found_bracket = False
    for scale in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]:
        if time.time() > deadline or eval_count >= max_evals:
            break
        
        h = scale * max(1.0, abs(x0))
        
        # Try left
        a_test = x0 - h
        fa_test = eval_f(a_test)
        if abs(fa_test) <= 1e-8:
            return a_test
        if fa_test * fa < 0:
            a, fa = a_test, fa_test
            b, fb = x0, fx
            found_bracket = True
            break
        
        # Try right
        b_test = x0 + h
        fb_test = eval_f(b_test)
        if abs(fb_test) <= 1e-8:
            return b_test
        if fa * fb_test < 0:
            a, fa = x0, fx
            b, fb = b_test, fb_test
            found_bracket = True
            break
    
    if found_bracket:
        # Bisection
        for i in range(60):
            if time.time() > deadline or eval_count >= max_evals:
                break
            
            mid = (a + b) / 2.0
            fmid = eval_f(mid)
            
            if abs(fmid) <= 1e-8:
                return mid
            
            if fmid * fa < 0:
                b = mid
                fb = fmid
            else:
                a = mid
                fa = fmid
            
            if abs(b - a) < 1e-14 * max(1.0, abs(a)):
                break
        
        # Return midpoint
        result = (a + b) / 2.0
        if abs(eval_f(result)) < abs(best_fx):
            best_x = result
            best_fx = eval_f(result)
    
    # Strategy 3: Nelder-Mead style local search to minimize |f(x)|
    if eval_count < max_evals and time.time() < deadline:
        # Simplex in 1D: just try small perturbations around best_x
        current_best = best_x
        current_best_val = abs(best_fx)
        
        for step_size in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]:
            if time.time() > deadline or eval_count >= max_evals:
                break
            
            improved = True
            while improved and eval_count < max_evals and time.time() < deadline:
                improved = False
                for dx in [step_size, -step_size]:
                    x_try = current_best + dx
                    fx_try = eval_f(x_try)
                    if abs(fx_try) < current_best_val:
                        current_best = x_try
                        current_best_val = abs(fx_try)
                        improved = True
                        if abs(fx_try) <= 1e-8:
                            return x_try
            step_size *= 0.5
    
    # Final polish: try to get closer with tight tolerance
    if abs(best_fx) <= 1e-8:
        return best_x
    
    # If we haven't achieved 1e-8, return the best we found
    # Try one more Newton-like step with careful derivative estimation
    if eval_count < max_evals and time.time() < deadline:
        h = 1e-8 * max(1.0, abs(best_x))
        fx_h = eval_f(best_x + h)
        if not math.isnan(fx_h):
            d = (fx_h - best_fx) / h
            if abs(d) > 1e-30:
                x_new = best_x - best_fx / d
                fx_new = eval_f(x_new)
                if abs(fx_new) < abs(best_fx):
                    best_x = x_new
                    best_fx = fx_new
    
    return best_x
# EVOLVE-BLOCK-END
