"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Derivative-free root finder using Inverse Quadratic Interpolation (IQI)
    with fallbacks to secant and bisection-like strategies.
    Designed for smooth polynomial roots.
    """
    deadline = time.time() + 110.0  # safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    calls = 1
    
    # If already close enough
    if abs(fx) <= 1e-8:
        return x
    
    # Try to find a bracket by expanding from x0
    # But first, try IQI/Secant which may converge fast without bracketing
    # We need at least 2 points for secant, 3 for IQI
    # Let's get a second point
    h = 1e-4 * max(1.0, abs(x))
    x2 = x + h
    fx2 = f(x2)
    calls += 1
    
    if abs(fx2) <= 1e-8:
        return x2
    
    # We have (x, fx) and (x2, fx2)
    # Maintain a history of the last 3 points for IQI
    # Points are (x_val, f_val)
    # We want to keep the points with smallest |f| to guide interpolation
    # But for IQI stability, we need distinct x values
    
    # Strategy: 
    # 1. Use Secant step if IQI not available
    # 2. Use IQI if 3 points available
    # 3. If steps are too large or diverging, try to bracket
    
    # Let's implement a robust loop
    # Keep a list of recent points, max 3
    # Actually, IQI works best with the 3 points closest to the root.
    # Let's keep a small history and always use the 3 points with smallest |f|
    # among the last few, ensuring distinct x.
    
    history = [(x, fx), (x2, fx2)]
    
    best_x = x if abs(fx) < abs(fx2) else x2
    best_fx = fx if abs(fx) < abs(fx2) else fx2
    
    for i in range(50):
        if time.time() > deadline:
            break
            
        # Check if we have a good enough root
        if abs(best_fx) <= 1e-8:
            return best_x
        
        # Determine next point
        next_x = None
        
        if len(history) >= 3:
            # Use IQI
            # Sort history by |f|, take top 3 distinct x
            # Actually, just use the last 3 distinct x values?
            # Better: use the 3 points with smallest |f| from history
            # But history might have duplicates or very close x's.
            # Let's just use the 3 most recent distinct points.
            # To keep it simple and robust, let's use the last 3 points in history.
            p1, p2, p3 = history[-3], history[-2], history[-1]
            x1, f1 = p1
            x2, f2 = p2
            x3, f3 = p3
            
            # Check for distinct x
            if x1 != x2 and x2 != x3 and x1 != x3:
                # Inverse Quadratic Interpolation
                # We want to find x such that f(x) = 0
                # Approximate x as a function of y = f(x)
                # Lagrange interpolation for x(y)
                # x(y) = x1 * ((y-f2)(y-f3)) / ((f1-f2)(f1-f3)) + ...
                # Set y = 0
                denom1 = (f1 - f2) * (f1 - f3)
                denom2 = (f2 - f1) * (f2 - f3)
                denom3 = (f3 - f1) * (f3 - f2)
                
                if abs(denom1) > 1e-30 and abs(denom2) > 1e-30 and abs(denom3) > 1e-30:
                    x_iqi = (x1 * f2 * f3 / denom1 + 
                             x2 * f1 * f3 / denom2 + 
                             x3 * f1 * f2 / denom3)
                    
                    # Sanity check: IQI step shouldn't be too far
                    # Compare distance to previous step
                    # If it's too large, fall back to secant
                    dist = abs(x_iqi - history[-1][0])
                    # Heuristic: if distance is > 10x the previous step size, use secant
                    # Or if it goes outside a reasonable range
                    if dist < 1e6 * max(1.0, abs(x)):
                        next_x = x_iqi
                    else:
                        next_x = None
                else:
                    next_x = None
            else:
                next_x = None
        
        if next_x is None:
            # Fall back to Secant
            if len(history) >= 2:
                x1, f1 = history[-2]
                x2, f2 = history[-1]
                if x1 != x2 and f1 != f2:
                    next_x = x2 - f2 * (x2 - x1) / (f2 - f1)
                else:
                    # Parallel or same point, nudge
                    next_x = x2 + 1e-4 * max(1.0, abs(x2))
            else:
                next_x = history[-1][0] + 1e-4 * max(1.0, abs(history[-1][0]))
        
        # Evaluate new point
        fx_new = f(next_x)
        calls += 1
        
        if not math.isfinite(fx_new):
            # If we got a non-finite value, shrink step or revert
            # For polynomials, this is rare unless x is huge
            # Let's just skip this point and try a smaller step
            continue
            
        if abs(fx_new) < abs(best_fx):
            best_x = next_x
            best_fx = fx_new
            
        # Update history
        # Keep only the last 3 points, but ensure we don't lose good points
        # A simple strategy: always add the new point, keep last 3.
        # However, if the new point is far away and worse, we might want to keep older better points.
        # Let's keep a history of up to 5 points, and use the 3 best (smallest |f|) for IQI?
        # No, IQI needs continuity. Let's just keep the last 3.
        # If we want to be smarter, we can replace the worst point in the history of 3.
        
        # Let's try: keep a list of up to 4 points.
        # For IQI, use the 3 with smallest |f| that are distinct.
        history.append((next_x, fx_new))
        if len(history) > 4:
            # Remove the worst point (largest |f|) to keep the set tight
            # But we must be careful not to remove points needed for interpolation stability.
            # Simpler: just keep last 3.
            history = history[-3:]
            
        # Check convergence
        if abs(fx_new) <= 1e-8:
            return next_x

    # If we didn't converge, return the best found
    # But we need to ensure |f(x)| <= 1e-8.
    # If best_fx is still > 1e-8, we might fail.
    # Let's do a final polish with Newton's method if possible, or just return best.
    # The problem states |f(x)| > 1e-8 is a failure.
    # So we must be careful.
    
    # If we haven't succeeded, try a few more secant steps with tighter tolerance
    # Or use the best_x and see if we can improve.
    
    # For now, return best_x. If it fails, the score will reflect it.
    # But we can try one more thing: if we have a bracket, use bisection.
    # We haven't been tracking brackets explicitly.
    
    # Let's check if we have a sign change in history
    # If so, bisect.
    if len(history) >= 2:
        # Find a sign change in the history
        for i in range(len(history) - 1):
            x1, f1 = history[i]
            x2, f2 = history[i+1]
            if f1 * f2 < 0:
                # We have a bracket
                a, fa = x1, f1
                b, fb = x2, f2
                # Bisection for a few steps
                for _ in range(50):
                    if time.time() > deadline:
                        break
                    m = (a + b) / 2
                    fm = f(m)
                    calls += 1
                    if abs(fm) <= 1e-8:
                        return m
                    if fa * fm < 0:
                        b = m
                        fb = fm
                    else:
                        a = m
                        fa = fm
                return (a + b) / 2

    return best_x
# EVOLVE-BLOCK-END
