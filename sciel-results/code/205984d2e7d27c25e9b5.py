"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust combination of:
    1. Secant method with adaptive step sizes
    2. Bisection fallback when sign change is detected
    3. Multiple restart strategies if initial attempts fail
    4. Deadline guard to ensure completion within time limit
    """
    x0 = float(x0)
    deadline = time.time() + 140.0  # Safety margin before 150s limit
    max_evals = 500
    eval_count = [0]
    
    def eval_f(x):
        if eval_count[0] >= max_evals:
            return None
        val = f(x)
        eval_count[0] += 1
        return val
    
    def check_time():
        return time.time() < deadline
    
    # Initial evaluation
    fx0 = eval_f(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= 1e-8:
        return x0
    
    # Try secant method first
    def secant_method(x_start, fx_start):
        if check_time():
            pass
        x_prev = x_start
        fx_prev = fx_start
        # Initial step for secant
        h = 1e-4 * max(1.0, abs(x_start))
        x_curr = x_start + h
        fx_curr = eval_f(x_curr)
        if fx_curr is None:
            return x_prev if abs(fx_prev) < abs(fx0) else x0
        if abs(fx_curr) <= 1e-8:
            return x_curr
        
        for _ in range(400):
            if not check_time():
                break
            if abs(fx_curr) <= 1e-8:
                return x_curr
            if abs(fx_prev) <= 1e-8:
                return x_prev
            
            # Secant step
            if fx_curr != fx_prev:
                x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
            else:
                break
            
            # Check for convergence
            if abs(x_new - x_curr) < 1e-12 * max(1.0, abs(x_curr)):
                fx_new = eval_f(x_new)
                if fx_new is not None and abs(fx_new) <= 1e-8:
                    return x_new
                break
            
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            # Update
            x_prev, fx_prev = x_curr, fx_curr
            x_curr, fx_curr = x_new, fx_new
        
        # Return best of the two most recent
        if abs(fx_curr) <= abs(fx_prev):
            return x_curr
        else:
            return x_prev
    
    # Try secant
    best_x = secant_method(x0, fx0)
    best_fx = eval_f(best_x) if eval_count[0] < max_evals else fx0
    if best_fx is not None and abs(best_fx) <= 1e-8:
        return best_x
    
    # Try bisection with bracket finding
    def find_bracket(x0, fx0):
        """Find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a, fa = x0, fx0
        # Try expanding bracket
        step = 1e-2 * max(1.0, abs(x0))
        for direction in [1, -1]:
            b = x0 + direction * step
            fb = eval_f(b)
            if fb is None:
                continue
            if fa * fb < 0:
                return a, b, fa, fb
            # Keep expanding
            for _ in range(20):
                if not check_time():
                    return None
                step *= 2
                b = x0 + direction * step
                fb = eval_f(b)
                if fb is None:
                    break
                if fa * fb < 0:
                    return a, b, fa, fb
        return None
    
    bracket = find_bracket(x0, fx0)
    if bracket is not None:
        a, b, fa, fb = bracket
        # Bisection
        for _ in range(60):
            if not check_time():
                break
            if eval_count[0] >= max_evals:
                break
            mid = (a + b) / 2.0
            fm = eval_f(mid)
            if fm is None:
                break
            if abs(fm) <= 1e-8:
                return mid
            if fa * fm < 0:
                b, fb = mid, fm
            else:
                a, fa = mid, fm
            if abs(b - a) < 1e-14:
                break
        # Return midpoint
        mid = (a + b) / 2.0
        fm = eval_f(mid)
        if fm is not None and abs(fm) <= 1e-8:
            return mid
        if abs(fm) < abs(best_fx if best_fx is not None else float('inf')):
            best_x = mid
            best_fx = fm
    
    # Try multiple restarts with different initial points
    restart_offsets = [1e-3, -1e-3, 1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0]
    for offset in restart_offsets:
        if not check_time():
            break
        if eval_count[0] >= max_evals:
            break
        x_try = x0 + offset * max(1.0, abs(x0))
        fx_try = eval_f(x_try)
        if fx_try is None:
            break
        if abs(fx_try) <= 1e-8:
            return x_try
        if abs(fx_try) < abs(best_fx if best_fx is not None else float('inf')):
            best_x = x_try
            best_fx = fx_try
        # Quick secant from this point
        x_res = secant_method(x_try, fx_try)
        fx_res = eval_f(x_res) if eval_count[0] < max_evals else None
        if fx_res is not None and abs(fx_res) <= 1e-8:
            return x_res
        if fx_res is not None and abs(fx_res) < abs(best_fx if best_fx is not None else float('inf')):
            best_x = x_res
            best_fx = fx_res
    
    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
