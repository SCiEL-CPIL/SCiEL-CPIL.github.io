"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach:
    1. Track evaluations and time budget.
    2. Try Newton's method with adaptive step and damping.
    3. Fall back to secant method.
    4. Use bisection if a bracket is found.
    5. Multiple restarts with different initial guesses if needed.
    """
    x0 = float(x0)
    start_time = time.time()
    DEADLINE = 140.0  # Leave 10s safety margin for 150s limit
    MAX_EVALS = 480   # Leave some margin under 500
    
    eval_count = [0]
    
    def eval_f(x):
        if eval_count[0] >= MAX_EVALS:
            raise ValueError("Max evaluations exceeded")
        if time.time() - start_time > DEADLINE:
            raise ValueError("Time limit exceeded")
        eval_count[0] += 1
        return f(x)
    
    best_x = x0
    best_f = abs(eval_f(x0))
    
    def update_best(x, fx):
        nonlocal best_x, best_f
        afx = abs(fx)
        if afx < best_f:
            best_x = x
            best_f = afx
        if afx <= 1e-8:
            return True
        return False
    
    # Check if x0 is already a root
    if best_f <= 1e-8:
        return best_x
    
    # Strategy 1: Newton's method with damping
    def newton(x_start, max_iter=50):
        x = float(x_start)
        fx = eval_f(x)
        update_best(x, fx)
        if abs(fx) <= 1e-8:
            return x
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            # Finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = eval_f(x + h)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            dfx = (fx_plus - fx) / h
            if abs(dfx) < 1e-15:
                break
            
            # Newton step
            x_new = x - fx / dfx
            
            # Damping: if step is too large, reduce it
            step = abs(x_new - x)
            if step > 1.0 * max(1.0, abs(x)):
                # Reduce step
                x_new = x + 0.5 * (x_new - x)
            
            # Check for divergence
            if not math.isfinite(x_new):
                break
            
            fx_new = eval_f(x_new)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if update_best(x_new, fx_new):
                return x_new
            
            # Accept step if it reduces |f|
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Try smaller step
                x_new = x - 0.5 * fx / dfx
                if not math.isfinite(x_new):
                    break
                fx_new = eval_f(x_new)
                if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                    break
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    break
        
        return x
    
    # Strategy 2: Secant method
    def secant(x0, x1, max_iter=50):
        x = float(x0)
        x_prev = float(x1)
        fx = eval_f(x)
        fx_prev = eval_f(x_prev)
        update_best(x, fx)
        update_best(x_prev, fx_prev)
        if abs(fx) <= 1e-8:
            return x
        if abs(fx_prev) <= 1e-8:
            return x_prev
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(fx - fx_prev) < 1e-15:
                break
            
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            
            x_prev = x
            fx_prev = fx
            x = x_new
            fx = eval_f(x)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if update_best(x, fx):
                return x
        
        return x
    
    # Strategy 3: Bisection (if we can find a bracket)
    def find_bracket(x0, max_expansions=20):
        """Try to find a bracket [a, b] where f(a) and f(b) have opposite signs."""
        a = float(x0)
        fa = eval_f(a)
        if abs(fa) <= 1e-8:
            return (a, a)
        
        # Try expanding in both directions
        step = 1.0
        for _ in range(max_expansions):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            # Try a + step
            b = a + step
            fb = eval_f(b)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            if abs(fb) <= 1e-8:
                return (b, b)
            if fa * fb < 0:
                return (a, b)
            
            # Try a - step
            c = a - step
            fc = eval_f(c)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            if abs(fc) <= 1e-8:
                return (c, c)
            if fa * fc < 0:
                return (c, a)
            
            step *= 2.0
            if step > 1e10:
                break
        
        return None
    
    def bisection(a, b, max_iter=60):
        """Bisection method on [a, b] where f(a)*f(b) < 0."""
        fa = eval_f(a)
        fb = eval_f(b)
        update_best(a, fa)
        update_best(b, fb)
        if abs(fa) <= 1e-8:
            return a
        if abs(fb) <= 1e-8:
            return b
        
        for i in range(max_iter):
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if abs(b - a) < 1e-14:
                break
            
            mid = 0.5 * (a + b)
            fm = eval_f(mid)
            if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
                break
            
            if update_best(mid, fm):
                return mid
            
            if fa * fm <= 0:
                b = mid
                fb = fm
            else:
                a = mid
                fa = fm
        
        return 0.5 * (a + b)
    
    # Try strategies in order
    # 1. Newton from x0
    x_result = newton(x0, max_iter=40)
    if best_f <= 1e-8:
        return best_x
    
    # 2. Secant from x0 and x0+small
    h = 1e-4 * max(1.0, abs(x0))
    x_result = secant(x0, x0 + h, max_iter=40)
    if best_f <= 1e-8:
        return best_x
    
    # 3. Try to find bracket and use bisection
    bracket = find_bracket(x0, max_expansions=15)
    if bracket is not None:
        a, b = bracket
        if a != b:
            x_result = bisection(a, b, max_iter=50)
            if best_f <= 1e-8:
                return best_x
    
    # 4. Try Newton from perturbed points
    for perturb in [1e-2, -1e-2, 1e-4, -1e-4, 1e-1, -1e-1]:
        if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
            break
        x_pert = x0 + perturb * max(1.0, abs(x0))
        x_result = newton(x_pert, max_iter=20)
        if best_f <= 1e-8:
            return best_x
    
    # 5. Try secant from perturbed points
    for perturb in [1e-3, -1e-3, 1e-5, -1e-5]:
        if time.time() - start_time > DEADLINE or eval_count[0] >= MAX_EVALS:
            break
        x_pert = x0 + perturb * max(1.0, abs(x0))
        x_result = secant(x0, x_pert, max_iter=20)
        if best_f <= 1e-8:
            return best_x
    
    # Return best result found
    return best_x
# EVOLVE-BLOCK-END
