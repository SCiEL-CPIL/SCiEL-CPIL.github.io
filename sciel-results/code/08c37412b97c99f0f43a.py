"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using inverse quadratic interpolation
    with secant fallback, targeting smooth polynomial equations.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    x = float(x0)
    
    # Initial evaluation
    fx = f(x)
    if abs(fx) <= 1e-10:
        return x
    
    # Start with 2 points for secant method
    h = 1e-4 * max(1.0, abs(x))
    x1 = x + h
    f1 = f(x1)
    
    if abs(f1) <= 1e-10:
        return x1
    
    best_x = x if abs(fx) < abs(f1) else x1
    best_fx = min(abs(fx), abs(f1))
    
    # Main iteration loop: inverse quadratic interpolation with safeguards
    # Keep track of the 3 most recent points for IQI
    for it in range(20):
        if time.time() > deadline:
            break
        
        # If we have 3 distinct points with distinct f-values, try IQI
        # Otherwise fall back to secant
        if it >= 2 and fx != f1 and fx != f2 and f1 != f2 and x != x1 and x != x2 and x1 != x2:
            # Inverse quadratic interpolation:
            # Find x such that f(x) = 0 using the 3 points (x,fx), (x1,f1), (x2,f2)
            # Formula: x_new = x*fx*(f1-f2)/(fx*(f1-f2)) + x1*f1*(f2-fx)/(f1*(f2-fx)) + x2*f2*(fx-f1)/(f2*(fx-f1))
            # Simplified: x_new = x + fx * [ (x1-x)*(f2-fx)/(fx*(f1-f2)) * (f1-fx)/(f1-fx) ... ]
            # Let's use the standard Lagrange form for inverse interpolation
            # We want to interpolate g(t) = x as a function of t = f(x)
            # g(0) using points (fx, x), (f1, x1), (f2, x2)
            
            # g(0) = x * (0-f1)*(0-f2) / ((fx-f1)*(fx-f2)) 
            #      + x1 * (0-fx)*(0-f2) / ((f1-fx)*(f1-f2))
            #      + x2 * (0-fx)*(0-f1) / ((f2-fx)*(f2-f1))
            
            denom1 = (fx - f1) * (fx - f2)
            denom2 = (f1 - fx) * (f1 - f2)
            denom3 = (f2 - fx) * (f2 - f1)
            
            if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                x_new = x * f1 * f2 / denom1 + x1 * fx * f2 / denom2 + x2 * fx * f1 / denom3
                
                if not math.isfinite(x_new):
                    # Fall back to secant
                    if fx == f1:
                        break
                    x_new = x - fx * (x1 - x) / (f1 - fx)
                    if not math.isfinite(x_new):
                        break
            else:
                # Fall back to secant
                if fx == f1:
                    break
                x_new = x - fx * (x1 - x) / (f1 - fx)
                if not math.isfinite(x_new):
                    break
        else:
            # Secant step
            if fx == f1:
                break
            x_new = x - fx * (x1 - x) / (f1 - fx)
            
            if not math.isfinite(x_new):
                break
        
        # Safeguard: don't go too far from current points
        x_ref = x1
        x_scale = max(1.0, abs(x_ref))
        if abs(x_new - x_ref) > 5.0 * x_scale:
            # Use bisection-like step instead
            x_new = (x + x1) / 2.0
        
        f_new = f(x_new)
        if abs(f_new) <= 1e-10:
            return x_new
        if abs(f_new) < best_fx:
            best_fx = abs(f_new)
            best_x = x_new
        
        # Update: shift points, keeping 3 most recent
        if it >= 2:
            x, fx = x1, f1
            x1, f1 = x2, f2
            x2, f2 = x_new, f_new
        else:
            x, fx = x1, f1
            x1, f1 = x_new, f_new
            if it == 0:
                x2, f2 = x_new, f_new  # duplicate for now
        
        # Check for convergence
        if abs(f_new) <= 1e-10:
            return x_new
    
    # Final high-precision polish: if we have a good candidate, try Newton-like
    # refinement using finite differences to get below 1e-8 more reliably
    if best_fx > 1e-8 and best_fx < 1e-6:
        # Try a few secant steps from the best point
        xp = best_x
        fp = f(xp)
        for _ in range(8):
            if time.time() > deadline:
                break
            if abs(fp) <= 1e-8:
                return xp
            # Estimate derivative
            dx = 1e-8 * max(1.0, abs(xp))
            fp2 = f(xp + dx)
            df = (fp2 - fp) / dx
            if abs(df) < 1e-30:
                break
            x_new = xp - fp / df
            if not math.isfinite(x_new):
                break
            f_new = f(x_new)
            if abs(f_new) <= 1e-8:
                return x_new
            if abs(f_new) < best_fx:
                best_fx = abs(f_new)
                best_x = x_new
            xp = x_new
            fp = f_new
    
    return best_x
# EVOLVE-BLOCK-END
