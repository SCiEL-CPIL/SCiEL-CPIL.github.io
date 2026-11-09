"""Root finding for ill-conditioned equations: robust adaptive hybrid."""
# EVOLVE-BLOCK-START
import time
import math
import sys

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using at most 500 evaluations of f.
    Returns the best x found (minimizing |f(x)|) within the budget.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    max_evals = 500
    evals_used = 0
    
    def check_time():
        return time.time() < deadline
    
    def safe_eval(x):
        nonlocal evals_used
        if not check_time():
            return None
        if evals_used >= max_evals:
            return None
        try:
            val = f(x)
            evals_used += 1
            if not isinstance(val, (int, float)):
                return None
            if math.isnan(val) or math.isinf(val):
                return None
            return float(val)
        except Exception:
            evals_used += 1
            return None

    # Track best point seen so far
    best_x = float(x0)
    best_fx = safe_eval(best_x)
    if best_fx is None:
        # Try a few simple perturbations if initial eval fails
        for dx in [1e-8, -1e-8, 1e-4, -1e-4, 1.0, -1.0]:
            if not check_time():
                break
            trial_x = float(x0) + dx
            trial_fx = safe_eval(trial_x)
            if trial_fx is not None:
                if abs(trial_fx) < abs(best_fx if best_fx is not None else float('inf')):
                    best_x = trial_x
                    best_fx = trial_fx
                if best_fx is not None and abs(best_fx) <= 1e-8:
                    return best_x
        if best_fx is None:
            return best_x
        if abs(best_fx) <= 1e-8:
            return best_x

    # If we already have a good solution, return it
    if abs(best_fx) <= 1e-8:
        return best_x

    # Phase 1: Safeguarded Newton with finite difference derivative
    # Use central differences when possible for better accuracy
    x = best_x
    fx = best_fx
    
    for it in range(30):
        if not check_time() or evals_used >= max_evals - 5:
            break
        if abs(fx) <= 1e-8:
            best_x = x
            best_fx = fx
            return x
            
        # Compute derivative using central difference if budget allows
        h = 1e-7 * max(1.0, abs(x))
        
        # Try central difference first
        if evals_used < max_evals - 2 and check_time():
            fp = safe_eval(x + h)
            fm = safe_eval(x - h)
            if fp is not None and fm is not None:
                df = (fp - fm) / (2 * h)
            else:
                # Fall back to forward difference
                if fp is not None:
                    df = (fp - fx) / h
                elif fm is not None:
                    df = (fx - fm) / h
                else:
                    break
        else:
            # Forward difference
            fp = safe_eval(x + h)
            if fp is None:
                break
            df = (fp - fx) / h
        
        if df is None or abs(df) < 1e-15:
            break
            
        # Newton step
        step = fx / df
        x_new = x - step
        
        # Safeguard: limit step size
        max_step = 10.0 * max(1.0, abs(x))
        if abs(step) > max_step:
            x_new = x - math.copysign(max_step, step)
        
        # Check if x_new is valid
        fx_new = safe_eval(x_new)
        if fx_new is None:
            # Try smaller step
            x_new = x - 0.5 * step
            fx_new = safe_eval(x_new)
            if fx_new is None:
                break
        
        # Update best
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
            
        if abs(fx_new) <= 1e-8:
            return x_new
        
        # Check for convergence
        if abs(step) < 1e-12 * max(1.0, abs(x_new)):
            break
            
        # Damping if function value didn't improve
        if abs(fx_new) > abs(fx) * 1.5:
            # Damp the step
            damp = 0.5
            x_new = x - damp * step
            fx_new = safe_eval(x_new)
            if fx_new is None or abs(fx_new) > abs(fx):
                break
        
        x = x_new
        fx = fx_new

    if abs(best_fx) <= 1e-8:
        return best_x

    # Phase 2: Bisection / Brent's method if we can bracket a root
    # Try to find a bracket around the current best point
    bracket_found = False
    a, b = None, None
    
    # Search for sign change in expanding intervals
    base_scale = max(1.0, abs(best_x))
    for scale in [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0, 1000.0]:
        if not check_time() or evals_used >= max_evals - 10:
            break
        delta = scale * base_scale
        xa = best_x - delta
        xb = best_x + delta
        
        fa = safe_eval(xa)
        if fa is None:
            continue
        fb = safe_eval(xb)
        if fb is None:
            continue
            
        if fa * fb < 0:
            a, b = xa, xb
            bracket_found = True
            break
        elif abs(fa) < abs(best_fx):
            best_x = xa
            best_fx = fa
            if abs(best_fx) <= 1e-8:
                return best_x
        elif abs(fb) < abs(best_fx):
            best_x = xb
            best_fx = fb
            if abs(best_fx) <= 1e-8:
                return best_x

    if bracket_found and check_time() and evals_used < max_evals - 5:
        # Use bisection (robust)
        # We could use scipy.optimize.brentq but let's implement bisection to be safe
        # and to control evaluation count precisely
        fa = safe_eval(a)
        fb = safe_eval(b)
        if fa is not None and fb is not None and fa * fb < 0:
            for _ in range(60):
                if not check_time() or evals_used >= max_evals - 2:
                    break
                mid = 0.5 * (a + b)
                fm = safe_eval(mid)
                if fm is None:
                    break
                if abs(fm) < abs(best_fx):
                    best_x = mid
                    best_fx = fm
                if abs(fm) <= 1e-8:
                    return mid
                if fa * fm < 0:
                    b = mid
                    fb = fm
                else:
                    a = mid
                    fa = fm
                if abs(b - a) < 1e-14 * max(1.0, abs(mid)):
                    break

    # Phase 3: Secant method as a backup if Newton failed
    if not check_time() or evals_used >= max_evals - 10:
        return best_x
        
    x0_sec = best_x
    x1_sec = best_x + 1e-6 * max(1.0, abs(best_x))
    fx0 = best_fx
    fx1 = safe_eval(x1_sec)
    if fx1 is None:
        return best_x
        
    if abs(fx1) < abs(best_fx):
        best_x = x1_sec
        best_fx = fx1
        if abs(best_fx) <= 1e-8:
            return best_x

    for _ in range(30):
        if not check_time() or evals_used >= max_evals - 2:
            break
        if abs(fx1) < abs(fx0):
            x0_sec, x1_sec = x1_sec, x0_sec
            fx0, fx1 = fx1, fx0
            
        if abs(fx1) <= 1e-8:
            return x1_sec
            
        if abs(fx1 - fx0) < 1e-15:
            break
            
        denom = fx1 - fx0
        if abs(denom) < 1e-15:
            break
            
        x2_sec = x1_sec - fx1 * (x1_sec - x0_sec) / denom
        
        # Safeguard
        if not (math.isfinite(x2_sec)):
            break
        max_step = 10.0 * max(1.0, abs(x1_sec))
        if abs(x2_sec - x1_sec) > max_step:
            x2_sec = x1_sec - math.copysign(max_step, x2_sec - x1_sec)
            
        fx2 = safe_eval(x2_sec)
        if fx2 is None:
            break
            
        if abs(fx2) < abs(best_fx):
            best_x = x2_sec
            best_fx = fx2
            if abs(best_fx) <= 1e-8:
                return best_x
                
        x0_sec, x1_sec = x1_sec, x2_sec
        fx0, fx1 = fx1, fx2

    return best_x
# EVOLVE-BLOCK-END
