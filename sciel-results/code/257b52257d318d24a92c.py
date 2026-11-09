"""Safeguarded Newton/Bisection hybrid for efficient root finding."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    x0 = float(x0)
    T0 = time.time()
    DEADLINE = 145.0          # Safety margin
    TOL = 1e-12               # Aim well below 1e-8
    HARD_TOL = 1e-8           # Contract tolerance
    MAX_CALLS = 450           # Hard limit on calls to ensure safety

    call_count = 0

    def elapsed_ok():
        return (time.time() - T0) < DEADLINE

    def safe_f(x):
        nonlocal call_count
        if call_count >= MAX_CALLS:
            return float('nan')
        try:
            val = f(x)
            call_count += 1
            if val != val or not math.isfinite(val):
                return float('nan')
            return val
        except Exception:
            call_count += 1
            return float('nan')

    # Evaluate start
    f0 = safe_f(x0)
    best_x = x0
    best_val = abs(f0) if f0 != float('nan') else float('inf')

    def track(x, fx):
        nonlocal best_x, best_val
        if fx != float('nan'):
            av = abs(fx)
            if av < best_val:
                best_val = av
                best_x = x

    track(x0, f0)

    if f0 != float('nan') and abs(f0) <= TOL:
        return x0

    # ------------------------------------------------------------------
    # 1) Find a bracket [a, b] with a sign change
    # ------------------------------------------------------------------
    a = b = None
    fa = fb = None

    # Start with a small step and expand geometrically
    # Try positive direction first, then negative
    base_step = max(1e-4, abs(x0) * 1e-4)
    
    # We will search outward from x0
    # To handle oscillatory functions, we need to be careful not to skip over roots.
    # However, with a budget of 500, we can't do a dense scan everywhere.
    # Geometric expansion is the standard robust approach.
    
    # Try to find a sign change in [x0, x0+step] and [x0-step, x0]
    # If not found, expand step size.
    
    # To save calls, we reuse evaluations.
    # Let's try a sequence of intervals:
    # [x0, x0+step], [x0-step, x0]
    # [x0+step, x0+2*step], [x0-2*step, x0-step]
    # etc.
    
    # Actually, simpler:
    # 1. Check x0 + step
    # 2. Check x0 - step
    # 3. If no sign change with x0, expand.
    
    # Let's use a list of candidate points to check, expanding outwards.
    # We keep the last evaluated point to avoid re-evaluation.
    
    # Initialize with x0
    # We want to find a, b such that f(a)*f(b) < 0.
    
    # Strategy:
    # 1. Try x0 + h
    # 2. If no sign change with x0, try x0 - h
    # 3. If no sign change, increase h and try further out.
    
    # To be efficient, we will maintain a "frontier" of evaluated points.
    
    # Let's just do a geometric expansion in both directions.
    # We will store evaluated points in a dict to avoid re-calling f.
    eval_cache = {x0: f0}
    
    def get_f(x):
        if x in eval_cache:
            return eval_cache[x]
        val = safe_f(x)
        eval_cache[x] = val
        track(x, val)
        return val

    # Initial step
    step = base_step
    
    # Try to find bracket
    # We will check intervals [x0, x0+step], [x0-step, x0]
    # Then [x0+step, x0+2*step], [x0-2*step, x0-step]
    
    # Let's iterate
    for _ in range(20): # Max expansion steps
        if not elapsed_ok() or call_count >= MAX_CALLS - 10:
            break
            
        # Positive side
        x_pos = x0 + step
        f_pos = get_f(x_pos)
        
        if f_pos != float('nan') and f0 != float('nan'):
            if f_pos * f0 < 0:
                a, b = x0, x_pos
                fa, fb = f0, f_pos
                break
            elif abs(f_pos) <= TOL:
                return x_pos
        
        # Negative side
        x_neg = x0 - step
        f_neg = get_f(x_neg)
        
        if f_neg != float('nan') and f0 != float('nan'):
            if f_neg * f0 < 0:
                a, b = x_neg, x0
                fa, fb = f_neg, f0
                break
            elif abs(f_neg) <= TOL:
                return x_neg
        
        # If we didn't find a sign change with x0, we need to look further out.
        # We need to check the interval between the last checked point and the next.
        # But we don't know the "last checked point" in a unified way if we alternate.
        # Let's keep track of the furthest positive and negative points checked.
        
        # Actually, the above loop is flawed because it doesn't check [x0+step, x0+2*step] if [x0, x0+step] didn't work.
        # Let's restructure.
        
        # Keep track of left and right frontiers
        # left_x, left_f, right_x, right_f
        # Initially: left_x = x0, right_x = x0
        
        # This is getting complex. Let's use a simpler, robust method:
        # Expand a window [L, R] centered at x0.
        # Start with L = x0 - step, R = x0 + step.
        # Evaluate f(L), f(R).
        # If sign change, bisect.
        # If not, expand L = L - (R-L), R = R + (R-L).
        
        # But we already evaluated x0.
        # Let's restart the bracketing logic properly.
        
        # Reset for this iteration's logic
        # We will use a dedicated bracketing function.
        
        # Break out to use a cleaner bracketing routine below
        pass

    # Clean Bracketing Routine
    a = b = None
    fa = fb = None
    
    # We have x0, f0.
    # Let's try to find a bracket by expanding from x0.
    # We'll check x0 + h, x0 - h, x0 + 2h, x0 - 2h, etc.
    # To ensure we don't miss a root between x0 and x0+h, we check that interval first.
    
    h = base_step
    last_pos_x = x0
    last_pos_f = f0
    last_neg_x = x0
    last_neg_f = f0
    
    for i in range(1, 30):
        if not elapsed_ok() or call_count >= MAX_CALLS - 10:
            break
            
        # Positive side
        x_new_pos = x0 + i * h
        f_new_pos = get_f(x_new_pos)
        
        if f_new_pos != float('nan') and last_pos_f != float('nan'):
            if f_new_pos * last_pos_f < 0:
                a, b = last_pos_x, x_new_pos
                fa, fb = last_pos_f, f_new_pos
                break
            elif abs(f_new_pos) <= TOL:
                return x_new_pos
        
        # Update frontier
        if f_new_pos != float('nan'):
            last_pos_x = x_new_pos
            last_pos_f = f_new_pos
            
        # Negative side
        x_new_neg = x0 - i * h
        f_new_neg = get_f(x_new_neg)
        
        if f_new_neg != float('nan') and last_neg_f != float('nan'):
            if f_new_neg * last_neg_f < 0:
                a, b = x_new_neg, last_neg_x
                fa, fb = f_new_neg, last_neg_f
                break
            elif abs(f_new_neg) <= TOL:
                return x_new_neg
        
        # Update frontier
        if f_new_neg != float('nan'):
            last_neg_x = x_new_neg
            last_neg_f = f_new_neg
        
        # Increase step size
        h *= 2.0

    # ------------------------------------------------------------------
    # 2) Refine with Safeguarded Newton
    # ------------------------------------------------------------------
    if a is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        x = 0.5 * (a + b)
        fx = get_f(x)
        
        if fx != float('nan') and abs(fx) <= TOL:
            return x
            
        # Safeguarded Newton Loop
        for _ in range(50):
            if not elapsed_ok() or call_count >= MAX_CALLS - 5:
                break
                
            if abs(fx) <= TOL:
                return x
                
            # Estimate derivative using central difference if possible, else forward
            # To save calls, we can use a small step relative to current bracket size
            # or a fixed small step.
            # Let's use a step based on the current interval width.
            dx = 1e-8 * max(1.0, abs(x))
            
            # Central difference: (f(x+dx) - f(x-dx)) / (2*dx)
            # This costs 2 calls.
            # Alternatively, use the slope from the previous point if we have it?
            # No, Newton needs local derivative.
            
            # Let's try to be smart: if we have a previous x and fx, we can use secant?
            # But we want Newton.
            
            # Let's just do forward difference to save 1 call?
            # f'(x) ~ (f(x+dx) - f(x)) / dx
            # Cost: 1 call.
            
            x_trial = x + dx
            f_trial = get_f(x_trial)
            
            if f_trial != float('nan'):
                df = (f_trial - fx) / dx
                
                if df != 0:
                    # Newton step
                    x_new = x - fx / df
                    
                    # Safeguard: if x_new is outside [a, b] or too far, use bisection
                    # Also, if x_new is not finite, use bisection
                    if math.isfinite(x_new) and a < x_new < b:
                        # Accept Newton step
                        x = x_new
                        fx = get_f(x)
                    else:
                        # Bisection
                        x = 0.5 * (a + b)
                        fx = get_f(x)
                else:
                    # Derivative is zero, use bisection
                    x = 0.5 * (a + b)
                    fx = get_f(x)
            else:
                # Evaluation failed, use bisection
                x = 0.5 * (a + b)
                fx = get_f(x)
                
            # Update bracket
            if fx != float('nan'):
                if fx * fa < 0:
                    b = x
                    fb = fx
                else:
                    a = x
                    fa = fx
                    
                if abs(fx) <= TOL:
                    return x

    # ------------------------------------------------------------------
    # 3) Return best candidate
    # ------------------------------------------------------------------
    if best_x == best_x and math.isfinite(best_val):
        return best_x
    return x0
# EVOLVE-BLOCK-END
