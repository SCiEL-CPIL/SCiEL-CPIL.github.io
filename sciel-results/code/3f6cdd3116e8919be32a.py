"""Root finding: Ill-conditioned - Adaptive Step-Doubling Bracketing with Modified Secant."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations.
    
    Strategy:
    1. Adaptive step-doubling to secure a sign-change bracket [a, b].
       If no bracket is found within a budget, fall back to the best point found.
    2. Modified Secant method within the bracket:
       - Use secant step x_new = a - f(a)*(b-a)/(f(b)-f(a)).
       - If x_new is outside [a,b] or doesn't reduce the interval sufficiently,
         shrink the larger sub-interval by a fixed factor (0.5) instead of bisection.
       - This handles ill-conditioned functions where standard secant might oscillate
         or converge slowly near the root.
    3. Track the best point (min |f(x)|) throughout and return it if convergence fails.
    """
    start_time = time.time()
    MAX_WALL = 118.0
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = 0
    
    def eval_f(x):
        nonlocal eval_count
        eval_count += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except:
            return float('inf')

    def time_ok():
        return time.time() - start_time < MAX_WALL

    x0 = float(x0)
    fx0 = eval_f(x0)
    
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= TARGET_TOL:
        return x0

    # --- Phase 1: Adaptive Step-Doubling Bracketing ---
    # We try to find a bracket [a, b] such that f(a)*f(b) < 0.
    # Start with a small step, double it until we find a sign change or exhaust budget.
    
    a, b, fa, fb = None, None, None, None
    found_bracket = False
    
    # Initial step size
    h = max(1e-6, abs(x0) * 1e-6)
    
    # Budget for bracketing: spend at most 30 evaluations
    bracket_budget = 30
    used_for_bracket = 0
    
    # First, check immediate neighbors
    for sign in [1, -1]:
        if used_for_bracket >= bracket_budget or not time_ok():
            break
        x_c = x0 + sign * h
        if x_c == x0:
            continue
        fx_c = eval_f(x_c)
        used_for_bracket += 1
        
        if abs(fx_c) <= TARGET_TOL:
            return x_c
        
        if abs(fx_c) < abs(best_fx):
            best_x, best_fx = x_c, fx_c
            
        if fx0 * fx_c < 0:
            # Found bracket
            if x0 < x_c:
                a, b, fa, fb = x0, x_c, fx0, fx_c
            else:
                a, b, fa, fb = x_c, x0, fx_c, fx0
            found_bracket = True
            break
    
    # If no bracket found with small step, try doubling
    if not found_bracket:
        h = max(1e-6, abs(x0) * 1e-6)
        for _ in range(20):  # Up to 20 doublings
            if used_for_bracket >= bracket_budget or not time_ok():
                break
            h *= 2
            if h > 1e12:
                break
                
            for sign in [1, -1]:
                if used_for_bracket >= bracket_budget or not time_ok():
                    break
                x_c = x0 + sign * h
                if x_c == x0:
                    continue
                fx_c = eval_f(x_c)
                used_for_bracket += 1
                
                if abs(fx_c) <= TARGET_TOL:
                    return x_c
                
                if abs(fx_c) < abs(best_fx):
                    best_x, best_fx = x_c, fx_c
                    
                if fx0 * fx_c < 0:
                    # Found bracket
                    if x0 < x_c:
                        a, b, fa, fb = x0, x_c, fx0, fx_c
                    else:
                        a, b, fa, fb = x_c, x0, fx_c, fx0
                    found_bracket = True
                    break
            if found_bracket:
                break

    # --- Phase 2: Modified Secant within Bracket ---
    if found_bracket:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Modified Secant:
        # We maintain [a, b] with f(a)*f(b) < 0.
        # At each step, compute secant point x_s = a - f(a)*(b-a)/(f(b)-f(a)).
        # If x_s is within [a, b], evaluate f(x_s).
        #   If f(a)*f(x_s) < 0, set b = x_s.
        #   Else, set a = x_s.
        # If x_s is outside [a, b] or the step doesn't reduce the interval by at least 50%,
        #   shrink the larger half of the interval by a factor of 0.5 (i.e., move the endpoint
        #   with the larger |f| towards the midpoint).
        
        while time_ok() and eval_count < MAX_EVALS:
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
            
            interval_len = b - a
            if interval_len <= 0:
                break
                
            # Compute secant point
            denom = fb - fa
            if denom == 0 or not math.isfinite(denom):
                # Fallback to bisection
                x_new = (a + b) / 2
            else:
                x_new = a - fa * (b - a) / denom
            
            # Check if secant point is within bracket and reduces interval
            use_secant = False
            if math.isfinite(x_new) and a < x_new < b:
                # Check if it reduces the interval by at least 50%
                new_len = min(abs(x_new - a), abs(b - x_new))
                if new_len < interval_len * 0.5:
                    use_secant = True
            
            if not use_secant:
                # Shrink the larger sub-interval by 0.5
                # If |fa| < |fb|, root is likely closer to a, so shrink right side
                if abs(fa) < abs(fb):
                    # Move b towards midpoint
                    x_new = b - 0.5 * interval_len
                else:
                    # Move a towards midpoint
                    x_new = a + 0.5 * interval_len
            
            fx_new = eval_f(x_new)
            
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            if abs(fx_new) < abs(best_fx):
                best_x, best_fx = x_new, fx_new
            
            # Update bracket
            if fa * fx_new < 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            # Update best
            if abs(fa) < abs(best_fx):
                best_x, best_fx = a, fa
            if abs(fb) < abs(best_fx):
                best_x, best_fx = b, fb

    # --- Phase 3: If no bracket, use best point or try local search ---
    # If we didn't find a bracket, we return the best point found.
    # For ill-conditioned problems, this might be sufficient if the function is smooth.
    
    return best_x

# EVOLVE-BLOCK-END
