"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Hybrid root finder: secant method with adaptive stepping, bisection fallback,
    and high-precision Newton polish. Returns x with |f(x)| <= 1e-8, minimizing
    function call count.
    """
    deadline = time.monotonic() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Check initial point
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    best_x = x
    best_f = fx
    
    # Try multiple step sizes to find a good initial pair for secant
    # Prefer steps that give a sign change (for bisection) or good secant behavior
    step_scales = [1e-2, 1e-1, 1.0, 10.0]
    x1 = x
    f1 = fx
    x2 = x
    f2 = fx
    found_sign_change = False
    
    for scale in step_scales:
        if time.monotonic() > deadline:
            break
        h = scale * max(1.0, abs(x))
        x_trial = x + h
        f_trial = f(x_trial)
        
        if abs(f_trial) <= 1e-8:
            return x_trial
        
        if abs(f_trial) < abs(best_f):
            best_x = x_trial
            best_f = f_trial
        
        # Check for sign change
        if fx * f_trial < 0:
            x2 = x_trial
            f2 = f_trial
            found_sign_change = True
            break
        
        # If no sign change, check if secant step from (x, x_trial) looks reasonable
        if f_trial != 0 and fx != 0:
            denom = f_trial - fx
            if denom != 0:
                x_secant = x_trial - f_trial * (x_trial - x) / denom
                if math.isfinite(x_secant):
                    x2 = x_trial
                    f2 = f_trial
                    break
    
    # If no good pair found, use the trial with smallest |f|
    if not found_sign_change and x2 == x:
        # Use the last trial
        h = step_scales[-1] * max(1.0, abs(x))
        x2 = x + h
        f2 = f(x2)
        if abs(f2) <= 1e-8:
            return x2
        if abs(f2) < abs(best_f):
            best_x = x2
            best_f = f2
    
    # Phase 1: Secant/Bisection hybrid
    for _ in range(50):
        if time.monotonic() > deadline:
            break
        
        # If we have a sign change, use bisection for guaranteed convergence
        if f1 * f2 < 0:
            x_new = (x1 + x2) / 2.0
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < abs(best_f):
                best_x = x_new
                best_f = f_new
            if f1 * f_new <= 0:
                x2, f2 = x_new, f_new
            else:
                x1, f1 = x_new, f_new
            # Check if we've converged enough to switch to polish
            if abs(f_new) < 1e-6:
                break
            continue
        
        # Secant step
        if f1 == f2:
            # Degenerate: try a different perturbation
            h = 1e-3 * max(1.0, abs(x1))
            x2 = x1 + h
            f2 = f(x2)
            if abs(f2) <= 1e-8:
                return x2
            if f1 == f2:
                break
            continue
        
        denom = f2 - f1
        if denom == 0:
            break
        
        x_new = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            # Fallback: nudge
            h = 1e-3 * max(1.0, abs(x2))
            x_new = x2 + h
        
        f_new = f(x_new)
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if abs(f_new) < abs(best_f):
            best_x = x_new
            best_f = f_new
        
        # Update: keep the two most recent points
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        
        # Check convergence
        if abs(x2 - x1) < 1e-12 * max(1.0, abs(x2)):
            break
        
        # If close enough, switch to polish phase
        if abs(f_new) < 1e-6:
            break
    
    # Phase 2: High-precision Newton polish
    # Use the best point found so far
    x = best_x
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x
    
    for _ in range(15):
        if time.monotonic() > deadline:
            break
        
        # Adaptive step for derivative estimation
        h = 1e-7 * max(1.0, abs(x))
        # Use central difference for better accuracy
        f_plus = f(x + h)
        f_minus = f(x - h)
        d = (f_plus - f_minus) / (2.0 * h)
        
        if d == 0 or not math.isfinite(d):
            # Fallback to forward difference
            d = (f_plus - fx) / h
            if d == 0 or not math.isfinite(d):
                break
        
        x_new = x - fx / d
        
        if not math.isfinite(x_new):
            break
        
        fx_new = f(x_new)
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        if abs(fx_new) < abs(best_f):
            best_x = x_new
            best_f = fx_new
        
        # Damped Newton if step is too large
        if abs(x_new - x) > 0.5 * max(1.0, abs(x)):
            x_new = x + 0.5 * (x_new - x)
            fx_new = f(x_new)
            if abs(fx_new) <= 1e-8:
                return x_new
        
        x = x_new
        fx = fx_new
        
        # Converged in x
        if abs(fx) < 1e-14:
            break
    
    return best_x
# EVOLVE-BLOCK-END
