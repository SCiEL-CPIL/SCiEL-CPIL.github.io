"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8 using at most 500 evaluations.
    Uses a robust hybrid approach:
    1. Secant method with adaptive steps
    2. Bisection fallback when sign change detected
    3. Nelder-Mead style local search for ill-conditioned cases
    4. Time and evaluation budget guards
    """
    x0 = float(x0)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    max_evals = 500
    eval_count = [0]
    
    def eval_f(x):
        if time.time() > deadline:
            return None
        if eval_count[0] >= max_evals:
            return None
        eval_count[0] += 1
        try:
            val = f(x)
            if math.isnan(val) or math.isinf(val):
                return None
            return val
        except (ValueError, OverflowError, ZeroDivisionError):
            return None
    
    # Try secant method first
    def try_secant(x0, budget=100):
        x = x0
        h = 1e-6 * max(1.0, abs(x))
        x1 = x + h
        
        fx0 = eval_f(x)
        if fx0 is None:
            return None
        if abs(fx0) <= 1e-8:
            return x
        
        fx1 = eval_f(x1)
        if fx1 is None:
            return None
        if abs(fx1) <= 1e-8:
            return x1
        
        for _ in range(budget):
            if time.time() > deadline:
                return None
            if fx0 is None or fx1 is None:
                break
            if abs(fx1) < abs(fx0):
                x, x1 = x1, x
                fx0, fx1 = fx1, fx0
            
            if abs(fx1) <= 1e-8:
                return x1
            
            # Check for sign change for bisection
            if fx0 * fx1 < 0:
                # Bisection
                for _ in range(60):
                    mid = 0.5 * (x + x1)
                    fmid = eval_f(mid)
                    if fmid is None:
                        break
                    if abs(fmid) <= 1e-8:
                        return mid
                    if fmid * fx0 < 0:
                        x1 = mid
                        fx1 = fmid
                    else:
                        x = mid
                        fx0 = fmid
                return x1 if abs(fx1) < abs(fx0) else x
            
            # Secant step
            if fx1 == fx0:
                # Degenerate, perturb
                x1 = x + 1e-8 * max(1.0, abs(x))
                fx1 = eval_f(x1)
                continue
            
            x_new = x1 - fx1 * (x1 - x) / (fx1 - fx0)
            
            # Damping if step is too large
            if abs(x_new - x1) > 10 * max(1.0, abs(x1)):
                x_new = x1 + 0.5 * (x1 - x)
            
            fx_new = eval_f(x_new)
            if fx_new is None:
                break
            
            if abs(fx_new) <= 1e-8:
                return x_new
            
            # Keep the two best points
            if abs(fx_new) < abs(fx0):
                x, fx0 = x1, fx1
                x1, fx1 = x_new, fx_new
            elif abs(fx_new) < abs(fx1):
                x1, fx1 = x_new, fx_new
            else:
                x, fx0 = x1, fx1
                x1, fx1 = x_new, fx_new
        
        # Return best found
        if fx0 is not None and fx1 is not None:
            return x if abs(fx0) < abs(fx1) else x1
        return x0
    
    # Try Nelder-Mead style minimization of f(x)^2
    def try_nm(x0, budget=150):
        # Simple 1D Nelder-Mead
        h = 1e-4 * max(1.0, abs(x0))
        simplex = [x0, x0 + h]
        fvals = [eval_f(x) for x in simplex]
        
        if any(fv is None for fv in fvals):
            return None
        
        best_x = simplex[0] if abs(fvals[0]) < abs(fvals[1]) else simplex[1]
        best_f = min(abs(fvals[0]), abs(fvals[1]))
        
        for _ in range(budget):
            if time.time() > deadline:
                break
            if best_f <= 1e-8:
                return best_x
            
            # Order: worst first
            if abs(fvals[0]) > abs(fvals[1]):
                simplex[0], simplex[1] = simplex[1], simplex[0]
                fvals[0], fvals[1] = fvals[1], fvals[0]
            
            x_worst = simplex[0]
            x_best = simplex[1]
            f_worst = fvals[0]
            f_best = fvals[1]
            
            # Reflection
            x_ref = x_best + (x_best - x_worst)
            f_ref = eval_f(x_ref)
            if f_ref is None:
                break
            
            if abs(f_ref) < abs(f_best):
                # Accept
                simplex[0] = x_ref
                fvals[0] = f_ref
                if abs(f_ref) <= 1e-8:
                    return x_ref
                # Try expansion
                x_exp = x_best + 2 * (x_ref - x_best)
                f_exp = eval_f(x_exp)
                if f_exp is not None and abs(f_exp) < abs(f_ref):
                    simplex[0] = x_exp
                    fvals[0] = f_exp
                    if abs(f_exp) <= 1e-8:
                        return x_exp
            else:
                # Contraction
                x_con = x_best + 0.5 * (x_worst - x_best)
                f_con = eval_f(x_con)
                if f_con is None:
                    break
                
                if abs(f_con) < abs(f_worst):
                    simplex[0] = x_con
                    fvals[0] = f_con
                else:
                    # Shrink
                    x_shrink = 0.5 * (x_best + x_worst)
                    f_shrink = eval_f(x_shrink)
                    if f_shrink is None:
                        break
                    simplex[0] = x_shrink
                    fvals[0] = f_shrink
            
            # Update best
            if abs(fvals[0]) < abs(fvals[1]):
                best_x = simplex[0]
                best_f = abs(fvals[0])
            else:
                best_x = simplex[1]
                best_f = abs(fvals[1])
        
        return best_x
    
    # Strategy: try secant first, then NM if needed
    result = try_secant(x0, budget=80)
    
    if result is not None:
        f_result = eval_f(result)
        if f_result is not None and abs(f_result) <= 1e-8:
            return result
    
    # Try NM
    result_nm = try_nm(x0, budget=150)
    if result_nm is not None:
        f_nm = eval_f(result_nm)
        if f_nm is not None and abs(f_nm) <= 1e-8:
            return result_nm
    
    # Return best available
    if result is not None:
        return result
    if result_nm is not None:
        return result_nm
    return x0
# EVOLVE-BLOCK-END
