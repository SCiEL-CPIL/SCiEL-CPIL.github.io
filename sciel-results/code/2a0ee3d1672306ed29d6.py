import numpy as np

def find_root(f, x0):
    x0 = float(x0)
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # Initialize Secant points
    # Use a small step to avoid division by zero or instability
    x1 = x0 + 1.0
    fx1 = f(x1)
    
    # If fx1 is also large, maybe expand or shrink?
    # Let's try a few initial steps to get a better direction
    
    x_curr = x0
    x_prev = x1
    f_curr = fx0
    f_prev = fx1
    
    # Maintain a bracket if we find one
    a, b = None, None
    fa, fb = None, None
    
    # Check if we have a bracket initially
    if fx0 * fx1 < 0:
        a, b = x0, x1
        fa, fb = fx0, fx1

    for i in range(50):
        # Check convergence
        if abs(f_curr) <= 1e-8:
            return x_curr
        if abs(f_prev) <= 1e-8:
            return x_prev

        # Calculate Secant step
        if abs(f_curr - f_prev) < 1e-15:
            # Parallel tangent, use a small step
            x_new = x_curr + 1e-4 * (1.0 + x_curr)
        else:
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        # Safeguard: If x_new is not finite or step is too large
        if not np.isfinite(x_new) or abs(x_new - x_curr) > 1e9:
            # Fallback to Bisection if we have a bracket
            if a is not None and b is not None:
                x_new = (a + b) / 2.0
            else:
                # Damped step
                x_new = x_curr - 0.5 * f_curr * (x_curr - x_prev) / (f_curr - f_prev) if abs(f_curr - f_prev) > 1e-15 else x_curr + 1.0

        f_new = f(x_new)
        
        if not np.isfinite(f_new):
            # Step was bad, shrink
            x_new = x_curr - 0.5 * (x_curr - x_prev)
            f_new = f(x_new)
            if not np.isfinite(f_new):
                # Give up on this path, try expanding bracket
                break

        # Update Bracket if possible
        if a is not None:
            if fa * f_new <= 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
        else:
            # Try to establish a bracket
            # Check if x_new and x_curr have opposite signs
            if f_new * f_curr < 0:
                a, fa = x_curr, f_curr
                b, fb = x_new, f_new
            elif f_new * f_prev < 0:
                a, fa = x_prev, f_prev
                b, fb = x_new, f_new

        # Decide which point to keep as x_curr
        # Strategy: Keep the point with smaller |f| as x_curr?
        # Or follow the Secant history?
        # Standard Secant: x_prev = x_curr, x_curr = x_new
        # But if we have a bracket, Bisection is safer.
        
        if a is not None and b is not None:
             # If we have a bracket, we can use Bisection to guarantee convergence
             # but Secant is faster. Let's use Secant if it stays in bracket, else Bisection.
             if a <= x_new <= b or b <= x_new <= a:
                 # Accept Secant step
                 pass
             else:
                 # Use Bisection step
                 x_new = (a + b) / 2.0
                 f_new = f(x_new)
                 # Update bracket
                 if fa * f_new <= 0:
                     b, fb = x_new, f_new
                 else:
                     a, fa = x_new, f_new

        # Update history
        # To ensure we don't lose the best point, we can track best_x
        # But Secant needs two distinct points.
        x_prev, x_curr = x_curr, x_new
        f_prev, f_curr = f_curr, f_new

        if abs(f_curr) <= 1e-8:
            return x_curr

    # Fallback: Return the best point found
    # We didn't store all points, but x_curr and x_prev are the last two.
    # Let's do a final check on a wider range if we failed
    # Actually, the loop runs 50 times. If it didn't converge, return the one with smallest |f|
    # We need to track the best point.
    
    # Let's restart with a slightly better structure that tracks best.
    pass
