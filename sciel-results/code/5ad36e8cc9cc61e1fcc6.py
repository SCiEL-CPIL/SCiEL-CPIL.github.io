import math
import time

_START_TIME = time.time()
_WALL_LIMIT = 115.0

def _time_left():
    return _WALL_LIMIT - (time.time() - _START_TIME)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x = float(x0)
    
    try:
        fx = f(x)
    except Exception:
        return x
        
    if not _is_finite(fx):
        return x
    if abs(fx) <= 1e-8:
        return x
    
    # Track best so far
    best_x = x
    best_fx = fx
    calls = 1

    # Initial second point
    scale = max(1.0, abs(x))
    # Use a slightly larger initial step to get a better slope estimate
    # but not so large we leave the basin of attraction
    x_prev = x - 1e-3 * scale 
    try:
        fx_prev = f(x_prev)
        calls += 1
    except Exception:
        return best_x
        
    if not _is_finite(fx_prev):
        return best_x
        
    if abs(fx_prev) < abs(best_fx):
        best_x = x_prev
        best_fx = fx_prev
    if abs(fx_prev) <= 1e-8:
        return x_prev

    # Main Loop: Secant with Bisection Fallback
    # We maintain a "window" [lo, hi] implicitly by tracking history
    # If Secant step increases |f|, we suspect divergence or oscillation
    # In that case, we bisect between x and x_prev
    
    for i in range(80):
        if _time_left() < 2.0:
            break
        
        # Check stagnation
        if abs(fx) < abs(fx_prev) and abs(fx) < 1e-15:
            # We are essentially at 0, but maybe not precise enough
            # Let it continue to polish
            pass

        # Calculate Secant Step
        denom = fx - fx_prev
        if abs(denom) < 1e-300:
            # Degenerate
            break
            
        x_secant = x - fx * (x - x_prev) / denom
        
        if not _is_finite(x_secant):
            break
        
        # Heuristic: Is the Secant step "too big" or increasing error?
        # We don't know the error without calling f(x_secant), but we can guess.
        # If x_secant is far away from x, it might be bad.
        # Let's just call f(x_secant) and see.
        
        try:
            fx_secant = f(x_secant)
            calls += 1
        except Exception:
            break
            
        if not _is_finite(fx_secant):
            break
            
        if abs(fx_secant) < abs(best_fx):
            best_x = x_secant
            best_fx = fx_secant
            
        if abs(fx_secant) <= 1e-8:
            return x_secant
        
        # Decide which point to keep for the next secant step
        # Standard Secant keeps (x, fx) and (x_new, fx_new)
        # But if fx_secant is WORSE than fx_prev, we might have jumped over the root 
        # or to a bad region.
        
        # Simple Strategy: 
        # If |f(x_secant)| < |f(x)|, move to x_secant.
        # If |f(x_secant)| >= |f(x)|, it's a bad step.
        # In a bad step, we can try Bisection between x and x_prev.
        
        if abs(fx_secant) < abs(fx):
            # Good step, proceed with Secant
            x_prev, fx_prev = x, fx
            x, fx = x_secant, fx_secant
        else:
            # Bad step. 
            # Let's try Bisection.
            # The root is likely between x and x_prev if they have opposite signs.
            # If they have same sign, bisection doesn't guarantee anything, 
            # but moving to the midpoint of the last two good points is safer than the secant jump.
            
            # Actually, if fx_secant is worse, we should probably revert 
            # to a smaller step. 
            # Let's use the bisection point between x and x_prev as the new candidate.
            x_bisect = (x + x_prev) / 2.0
            try:
                fx_bisect = f(x_bisect)
                calls += 1
            except Exception:
                break
                
            if not _is_finite(fx_bisect):
                break
                
            if abs(fx_bisect) < abs(best_fx):
                best_x = x_bisect
                best_fx = fx_bisect
                
            if abs(fx_bisect) <= 1e-8:
                return x_bisect
            
            # Update state: 
            # We now have (x_prev, fx_prev) and (x, fx).
            # The bisection point is between them.
            # For the next Secant, we should probably use (x_prev, fx_prev) and (x_bisect, fx_bisect)
            # or (x, fx) and (x_bisect, fx_bisect).
            # Let's keep the one with the smaller |f| as the anchor.
            
            if abs(fx_bisect) < abs(fx):
                # Use x_bisect as the new x
                x, fx = x_bisect, fx_bisect
            else:
                # Use x_prev as the new x? No, x_prev is older.
                # Just stick with x and x_prev?
                # If we do nothing, we will repeat the same Secant step?
                # No, because we didn't update x_prev.
                # If we don't update, we might loop.
                # Let's update x_prev to x, and x to x_bisect?
                # This ensures progress towards the center.
                x_prev, fx_prev = x, fx
                x, fx = x_bisect, fx_bisect
            
            # If we are oscillating, check convergence of the interval
            if abs(x - x_prev) < 1e-14 * max(1.0, abs(x)):
                 break

    # Polish Phase: Newton with Central Difference
    # Start from best_x
    x = best_x
    fx = best_fx
    
    for i in range(10):
        if _time_left() < 1.0:
            break
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
            fx_minus = f(x - h)
            calls += 2
        except Exception:
            break
            
        if not _is_finite(fx_plus) or not _is_finite(fx_minus):
            break
            
        d = (fx_plus - fx_minus) / (2.0 * h)
        
        if d == 0 or not _is_finite(d):
            break
            
        x_new = x - fx / d
        if not _is_finite(x_new):
            break
            
        try:
            fx_new = f(x_new)
            calls += 1
        except Exception:
            break
            
        if not _is_finite(fx_new):
            break
            
        if abs(fx_new) < abs(best_fx):
            best_x = x_new
            best_fx = fx_new
            
        if abs(fx_new) <= 1e-8:
            return x_new
            
        x = x_new
        fx = fx_new
        
        if abs(x_new - x) < 1e-15:
            break
            
    return best_x
