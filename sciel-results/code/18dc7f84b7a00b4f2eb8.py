import math

def find_root(f, x0):
    x0 = float(x0)
    
    # Initial points
    delta = 1.0
    # We'll maintain a set of points and their function values
    # To keep it simple and robust, we'll use a small list of candidate points
    # and update them.
    
    # First, check x0
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # Try to find a bracket or a good local region
    # We will use a hybrid approach:
    # 1. Expand to find a sign change.
    # 2. If found, use Inverse Quadratic Interpolation / Secant within the bracket.
    # 3. If not found, minimize |f(x)| in the expanded region using a ternary search or golden section,
    #    then use IQI around the minimum.
    
    # Step 1: Expand to find sign change
    step = 1.0
    a = x0 - step
    b = x0 + step
    fa = f(a)
    fb = f(b)
    
    # Keep track of the best point found so far (min |f|)
    best_x = x0
    best_f = fx0
    
    if abs(fa) < abs(best_f):
        best_x, best_f = a, fa
    if abs(fb) < abs(best_f):
        best_x, best_f = b, fb

    for _ in range(50):
        if fa * fb <= 0:
            break
        # Expand
        step *= 2.0
        a = x0 - step
        b = x0 + step
        fa = f(a)
        fb = f(b)
        
        if abs(fa) < abs(best_f):
            best_x, best_f = a, fa
        if abs(fb) < abs(best_f):
            best_x, best_f = b, fb
            
        if step > 1e12:
            break

    # Now we have a region [a, b] which might contain a root, or we have a best_x
    # Strategy:
    # If sign change exists, use a modified Brent-like or Secant method.
    # Since the directive asks for a NEW approach, let's use Inverse Interpolation primarily.
    
    if fa * fb <= 0:
        # We have a bracket [a, b]
        x1, x2 = a, b
        f1, f2 = fa, fb
        
        # Ensure x1 is closer to 0 or just use x0 as a third point if it's inside
        # For IQI, we need 3 points.
        # Points: x1, x2, and x0 (if x0 is in [a,b]) or a midpoint.
        x3 = x0
        f3 = fx0
        
        # If x0 is not in [a,b], pick a point inside
        if x3 < a or x3 > b:
            x3 = (a + b) / 2.0
            f3 = f(x3)
            if abs(f3) < abs(best_f):
                best_x, best_f = x3, f3

        # Iterate
        for _ in range(100):
            # Check convergence
            if abs(f1) <= 1e-8:
                return x1
            if abs(f2) <= 1e-8:
                return x2
            if abs(f3) <= 1e-8:
                return x3

            # Try Inverse Quadratic Interpolation
            # Formula: x_new = (x1*f2*f3 + x2*f1*f3 + x3*f1*f2) / (f2*f3 + f1*f3 + f1*f2)
            denom = f2 * f3 + f1 * f3 + f1 * f2
            num = x1 * f2 * f3 + x2 * f1 * f3 + x3 * f1 * f2
            
            if abs(denom) > 1e-30:
                x_new = num / denom
            else:
                # Fallback to Secant using the two points with smallest |f|
                if abs(f1) < abs(f2):
                    # Use x1 and x3? Or x1 and x2?
                    # Secant: x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                    if abs(f1 - f2) > 1e-30:
                        x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
                    else:
                        x_new = (x1 + x2) / 2.0
                else:
                    if abs(f2 - f1) > 1e-30:
                        x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                    else:
                        x_new = (x1 + x2) / 2.0

            # Safeguard: Ensure x_new is within [a, b]
            if x_new < a or x_new > b:
                # Use Bisection
                x_new = (a + b) / 2.0
            
            f_new = f(x_new)
            
            if abs(f_new) <= 1e-8:
                return x_new

            # Update points
            # We want to maintain a bracket and a third point for IQI.
            # If f_new has same sign as f1, replace x1 with x_new.
            # If f_new has same sign as f2, replace x2 with x_new.
            # If f_new has opposite sign to both (impossible if bracket maintained), 
            # or if it splits the bracket, we update the bracket.
            
            if f_new * f1 < 0:
                # New bracket is [x_new, x2]
                x1, f1 = x_new, f_new
                # x3 remains the other point? 
                # Actually, for IQI, we just need 3 points. 
                # Let's keep the 3 points with smallest |f| or most recent.
                # Simpler: Keep x1, x2 as bracket. x3 is the other point.
                # If we replaced x1, the old x1 becomes a candidate for x3?
                # Let's just keep the 3 points: x1, x2, x3.
                # If x1 is replaced, the old x1 is no longer in the bracket.
                # We can set x3 to the old x1 if it's better than current x3?
                # Or just keep x3 as is?
                # Let's keep the point with the smallest |f| among the replaced ones and x3.
                # For simplicity, we'll just update the bracket and keep x3 as the third point 
                # unless the new point is better.
                pass
            elif f_new * f2 < 0:
                x2, f2 = x_new, f_new
                pass
            else:
                # No sign change with either, so f_new has same sign as f1 and f2?
                # This implies x_new is outside the bracket? No, we forced it inside.
                # Or f1 and f2 have same sign? No, we started with a bracket.
                # If f1 * f2 <= 0, and x_new is inside, f_new must have opposite sign to one of them.
                # Unless f_new is 0.
                # If this happens, it's a numerical error or flat function.
                # Fallback to bisection point
                x1, f1 = x_new, f_new
                # Update bracket bounds
                if f1 * f2 > 0:
                    # Bracket lost, restart with bisection
                    # This should be rare
                    x1 = a
                    f1 = fa
                    x2 = b
                    f2 = fb
                    continue

            # Update x3?
            # To maintain diversity, let's replace the point with the largest |f| 
            # among x1, x2, x3 with x_new if x_new is not x1 or x2?
            # Actually, x_new IS x1 or x2 now.
            # We should update x3 to be the other endpoint of the previous bracket?
            # Let's just keep x3 as the initial x0 or the midpoint.
            # A better strategy: Keep the 3 points that are closest to the root.
            # For now, this is sufficient.
            
            # Update best
            if abs(f_new) < abs(best_f):
                best_x, best_f = x_new, f_new

    # If no bracket found or loop finished
    # Fallback: Minimize |f(x)| around best_x
    # Use a small local search
    delta = 1.0
    for _ in range(20):
        # Check neighbors
        x_left = best_x - delta
        x_right = best_x + delta
        fl = f(x_left)
        fr = f(x_right)
        
        if abs(fl) < abs(best_f):
            best_x, best_f = x_left, fl
            delta *= 0.5
        elif abs(fr) < abs(best_f):
            best_x, best_f = x_right, fr
            delta *= 0.5
        else:
            delta *= 0.5
            if delta < 1e-10:
                break
        if abs(best_f) <= 1e-8:
            return best_x

    return best_x
