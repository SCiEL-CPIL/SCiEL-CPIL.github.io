"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Strategy:
    1. Check x0.
    2. Adaptive geometric expansion to find a bracket [a, b] with sign change.
    3. Use Illinois/secant hybrid within the bracket for fast convergence.
    4. Deadline guard for safety.
    """
    tol = 1e-8
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin for 120s limit
    
    def time_up():
        return time.time() > deadline

    x = float(x0)
    fx = f(x)
    
    if not math.isfinite(fx):
        # If f(x0) is not finite, try moving away
        for dx in [1.0, -1.0, 10.0, -10.0, 100.0, -100.0]:
            x_try = x + dx
            fx_try = f(x_try)
            if math.isfinite(fx_try):
                x = x_try
                fx = fx_try
                break
        else:
            return x  # Return x0 as best effort

    if abs(fx) <= tol:
        return x

    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # Strategy A: Adaptive geometric expansion to find a bracket
    # We look for a sign change f(a)*f(b) < 0
    a = x
    fa = fx
    b = None
    fb = None
    
    # Try expanding in both directions with geometric steps
    # Start with small steps, increase by factor of 2
    step = 1e-4
    max_step = 1e8
    found_bracket = False
    
    while step <= max_step and not time_up():
        # Try forward expansion
        x_plus = a + step
        fx_plus = f(x_plus)
        
        if math.isfinite(fx_plus):
            if abs(fx_plus) <= tol:
                return x_plus
            if fx_plus * fa < 0:
                b = x_plus
                fb = fx_plus
                found_bracket = True
                break
        
        # Try backward expansion
        x_minus = a - step
        fx_minus = f(x_minus)
        
        if math.isfinite(fx_minus):
            if abs(fx_minus) <= tol:
                return x_minus
            if fx_minus * fa < 0:
                b = x_minus
                fb = fx_minus
                found_bracket = True
                break
        
        # If we found a point with smaller |f|, update best
        if math.isfinite(fx_plus) and abs(fx_plus) < best_abs_fx:
            best_x = x_plus
            best_fx = fx_plus
            best_abs_fx = abs(fx_plus)
        if math.isfinite(fx_minus) and abs(fx_minus) < best_abs_fx:
            best_x = x_minus
            best_fx = fx_minus
            best_abs_fx = abs(fx_minus)
            
        step *= 2.0

    if not found_bracket:
        # No bracket found via simple expansion.
        # Try a denser local scan around x0 or best_x
        center = best_x
        fc = best_fx
        scan_step = 1e-3
        max_scan_step = 10.0
        
        # Scan in a window around center
        n_points = 100
        # Use numpy for vectorized evaluation if possible, but f takes float.
        # So we do a loop but with numpy for index generation
        xs = np.linspace(center - max_scan_step, center + max_scan_step, n_points)
        
        for i in range(len(xs)):
            if time_up():
                break
            xi = float(xs[i])
            fxi = f(xi)
            if math.isfinite(fxi):
                if abs(fxi) <= tol:
                    return xi
                if fxi * fc < 0:
                    a = center
                    b = xi
                    fa = fc
                    fb = fxi
                    found_bracket = True
                    break
                if abs(fxi) < best_abs_fx:
                    best_x = xi
                    best_fx = fxi
                    best_abs_fx = abs(fxi)
        
        if not found_bracket:
            # Return best found
            return best_x

    # We have a bracket [a, b] with fa*fb < 0
    # Use Illinois method (modified bisection) for robustness and speed
    # Illinois converges faster than bisection and is robust like bisection
    
    x1, f1 = a, fa
    x2, f2 = b, fb
    
    # Ensure f1 < 0 < f2 or f1 > 0 > f2 for consistent handling
    # Actually, Illinois method works regardless of sign, just need f1*f2 < 0
    
    for i in range(100):
        if time_up():
            break
            
        # Illinois step
        # If |f1| < |f2|, swap to make f1 the smaller magnitude
        if abs(f1) < abs(f2):
            x1, x2 = x2, x1
            f1, f2 = f2, f1
            
        # Illinois: x_new = x1 + (x2 - x1) * f1 / (f1 - f2)
        # But with damping: if f1 and f2 have same sign (shouldn't happen), fall back to bisection
        # Standard Illinois:
        # x_new = x1 + (x2 - x1) * f1 / (f1 - f2)
        # But if this goes out of bounds, use bisection
        denom = f1 - f2
        if denom == 0:
            x_new = (x1 + x2) / 2.0
        else:
            x_new = x1 + (x2 - x1) * f1 / denom
            
        # Check if x_new is within [min(x1,x2), max(x1,x2)]
        if x_new < min(x1, x2) or x_new > max(x1, x2):
            x_new = (x1 + x2) / 2.0
            
        f_new = f(x_new)
        
        if not math.isfinite(f_new):
            # Fallback to bisection
            x_new = (x1 + x2) / 2.0
            f_new = f(x_new)
            if not math.isfinite(f_new):
                break
                
        if abs(f_new) <= tol:
            return x_new
            
        # Update bracket
        if f_new * f1 < 0:
            x2 = x_new
            f2 = f_new
        else:
            # f_new and f2 have same sign, so root is between x1 and x_new
            # Illinois: damp f2
            x1 = x_new
            f1 = f_new
            f2 *= 0.5  # Illinois damping
            
        # Convergence check
        if abs(x2 - x1) < 1e-14 * max(1.0, abs(x2)):
            # Return the one with smaller |f|
            if abs(f1) < abs(f2):
                return x1
            else:
                return x2
                
    # Return the point with the smallest |f|
    if abs(f1) < abs(f2):
        return x1
    else:
        return x2

# EVOLVE-BLOCK-END
