"""Derivative-free trust-region root finder for ill-conditioned 1D problems."""
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using derivative-free trust-region minimization.
    Treats root-finding as minimizing |f(x)| directly with adaptive step sizes.
    
    Budget: at most 500 evaluations of f.
    """
    target_tol = 1e-8
    max_evals = 500
    eval_count = [0]

    def eval_f(x):
        eval_count[0] += 1
        try:
            val = f(x)
            if not (isinstance(val, (int, float)) or (hasattr(val, '__float__'))):
                val = float(val)
            if math.isnan(val) or math.isinf(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')

    def abs_f(x):
        return abs(eval_f(x))

    # Evaluate at x0
    x = float(x0)
    f0 = eval_f(x)
    if abs(f0) <= target_tol:
        return x

    # --- Phase 1: Aggressive bracketing search ---
    bracket_found = False
    a, b = None, None
    fa, fb = None, None

    # Try expanding a bracket with geometric progression
    step = max(1e-6, 1e-6 * abs(x))
    best_x, best_f = x, abs(f0)
    
    for _ in range(30):
        if eval_count[0] >= max_evals - 50:
            break
        
        x1 = x + step
        x2 = x - step
        f1 = eval_f(x1)
        f2 = eval_f(x2)
        
        if abs(f1) <= target_tol:
            return x1
        if abs(f2) <= target_tol:
            return x2
        
        if abs(f1) < best_f:
            best_x, best_f = x1, abs(f1)
        if abs(f2) < best_f:
            best_x, best_f = x2, abs(f2)
        
        # Check for sign changes
        if f0 * f1 < 0:
            a, b = x, x1
            fa, fb = f0, f1
            bracket_found = True
            break
        if f0 * f2 < 0:
            a, b = x2, x
            fa, fb = f2, f0
            bracket_found = True
            break
        if f1 * f2 < 0:
            a, b = x1, x2
            fa, fb = f1, f2
            bracket_found = True
            break
        
        step *= 3.0
        if step > 1e12:
            break

    # --- Phase 2: If bracket found, use robust bisection + interpolation ---
    if bracket_found:
        # Ensure |fa| < |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c = a
        fc = fa
        
        for _ in range(100):
            if eval_count[0] >= max_evals - 5:
                break
            
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                break
            
            # Try interpolation first
            x_new = None
            if abs(fc) > abs(fa):
                # Inverse quadratic interpolation
                num = fa * fc * (fb - fa)
                den = fa * fb * (fc - fa)
                if abs(den) > 1e-30:
                    x_new = a + num / den
            
            # Check if interpolation is safe
            if x_new is not None and a < x_new < b:
                f_new = eval_f(x_new)
                if abs(f_new) <= target_tol:
                    return x_new
                
                # Update bracket
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                
                # Update c
                if abs(f_new) < abs(fc):
                    c, fc = x_new, f_new
            else:
                # Bisection
                x_new = 0.5 * (a + b)
                f_new = eval_f(x_new)
                if abs(f_new) <= target_tol:
                    return x_new
                
                if fa * f_new < 0:
                    b, fb = x_new, f_new
                else:
                    a, fa = x_new, f_new
                
                if abs(f_new) < abs(fc):
                    c, fc = x_new, f_new
        
        # Return the endpoint with smaller |f|
        if abs(fa) < abs(fb):
            return a
        return b

    # --- Phase 3: No bracket. Use adaptive trust-region minimization of |f(x)| ---
    # Start with best point found so far
    best_x = best_x
    best_f = best_f
    
    # Adaptive step size for local search
    step = max(1e-4, 1e-4 * abs(best_x))
    no_improve = 0
    max_no_improve = 15
    
    for i in range(min(200, max_evals - eval_count[0] - 20)):
        if best_f <= target_tol:
            break
        if eval_count[0] >= max_evals - 10:
            break
        
        # Try multiple candidate points
        candidates = [
            best_x + step,
            best_x - step,
            best_x + step * 0.5,
            best_x - step * 0.5,
            best_x + step * 2.0,
            best_x - step * 2.0
        ]
        
        best_candidate = None
        best_candidate_f = best_f
        
        for cand in candidates:
            if eval_count[0] >= max_evals - 5:
                break
            cf = abs_f(cand)
            if cf < best_candidate_f:
                best_candidate_f = cf
                best_candidate = cand
        
        if best_candidate is not None:
            best_x = best_candidate
            best_f = best_candidate_f
            no_improve = 0
            # Reduce step if we made good progress
            if best_f < best_candidate_f * 0.5:
                step *= 0.7
        else:
            no_improve += 1
            # Increase step if no improvement
            step *= 1.5
            if no_improve >= max_no_improve or step > 1e10:
                break
    
    if best_f <= target_tol:
        return best_x

    # --- Phase 4: Final secant refinement ---
    # Try secant method from best point with a nearby point
    x_a = best_x
    f_a = eval_f(x_a)
    
    # Find a second point
    x_b = x_a + step
    f_b = eval_f(x_b)
    
    for i in range(min(50, max_evals - eval_count[0] - 2)):
        if eval_count[0] >= max_evals - 2:
            break
        
        if abs(f_b - f_a) < 1e-30:
            break
        
        x_new = x_b - f_b * (x_b - x_a) / (f_b - f_a)
        
        # Safeguard
        if abs(x_new - x_b) > 10 * abs(x_b - x_a):
            x_new = x_b + 0.5 * (x_b - x_a)
        
        f_new = eval_f(x_new)
        if abs(f_new) <= target_tol:
            return x_new
        
        x_a, f_a = x_b, f_b
        x_b, f_b = x_new, f_new
    
    # Return the best point
    if abs(f_a) < abs(f_b):
        return x_a
    return x_b

# EVOLVE-BLOCK-END
