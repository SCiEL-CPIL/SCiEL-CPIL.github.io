import math

def find_root(f, x0):
    x0 = float(x0)
    tol = 1e-8
    
    # 1. Try to find a bracket [a, b] such that f(a)*f(b) < 0
    # Start with a small step size
    h = 1e-4 * max(1.0, abs(x0))
    
    # Evaluate f at x0
    fx0 = f(x0)
    if abs(fx0) <= tol:
        return x0
    
    # We will expand the search window symmetrically
    # Keep track of the best point found so far (min |f(x)|)
    best_x = x0
    best_f = fx0
    best_abs_f = abs(fx0)
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Expand up to 50 times (h grows exponentially)
    # This covers a huge range if x0 is far from root
    for i in range(50):
        x_left = x0 - h
        x_right = x0 + h
        
        fx_left = f(x_left)
        if abs(fx_left) < best_abs_f:
            best_abs_f = abs(fx_left)
            best_x = x_left
            best_f = fx_left
        if abs(fx_left) <= tol:
            return x_left
            
        fx_right = f(x_right)
        if abs(fx_right) < best_abs_f:
            best_abs_f = abs(fx_right)
            best_x = x_right
            best_f = fx_right
        if abs(fx_right) <= tol:
            return x_right
            
        # Check for sign changes
        # Between x_left and x0
        if fx_left * fx0 < 0:
            a = x_left
            b = x0
            fa = fx_left
            fb = fx0
            break
        # Between x0 and x_right
        if fx0 * fx_right < 0:
            a = x0
            b = x_right
            fa = fx0
            fb = fx_right
            break
            
        # If no sign change, expand h
        h *= 2.0
        
        # Optional: Check if we are going too far or if we missed a root?
        # For oscillatory functions, we might jump over a root if h is too big.
        # However, we start small and double. 
        # If the frequency is very high, we might miss it.
        # But x0 is likely a "starting point" close to a root.
        # If we don't find a bracket, we return the best guess.
        
    if a is None:
        # No bracket found. Return best point found.
        # This might fail the tolerance check, but it's the best we can do without more info.
        return best_x

    # 2. Refine the root using Illinois Algorithm
    # Illinois is a variant of Bisection that uses Secant steps but shrinks the 
    # larger sub-interval's function value to maintain linear convergence.
    
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
        
    for _ in range(100): # Max iterations
        # Secant step
        # c = (a*fb - b*fa) / (fb - fa)
        denom = fb - fa
        if denom == 0:
            # Degenerate case, fallback to bisection
            c = (a + b) / 2.0
        else:
            c = (a * fb - b * fa) / denom
        
        fc = f(c)
        if abs(fc) < best_abs_f:
            best_abs_f = abs(fc)
            best_x = c
            best_f = fc
            
        if abs(fc) <= tol:
            return c
            
        # Update bracket
        if fa * fc < 0:
            # Root is in [a, c]
            b = c
            fb = fc
        else:
            # Root is in [c, b]
            a = c
            fa = fc
            
        # Illinois modification:
        # Shrink the function value of the endpoint that is further from c
        if (b - c) < (c - a):
            # The left interval is smaller, so shrink the right side's influence
            # by halving fb
            fb /= 2.0
        else:
            # The right interval is smaller, so shrink the left side's influence
            # by halving fa
            fa /= 2.0
            
        # Check for convergence based on interval width
        if abs(b - a) < 1e-14:
            break
            
    return best_x
