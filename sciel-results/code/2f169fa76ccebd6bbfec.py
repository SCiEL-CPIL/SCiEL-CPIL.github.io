import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Robust root finder using a hybrid bisection-secant method with an initial
    bracketing phase. Optimized for smooth polynomial functions to minimize
    the number of function calls.
    """
    deadline = time.time() + 110.0
    tol = 1e-10  # Target tighter than 1e-8 for safety
    max_calls = 450  # Safety margin below 500
    calls = 0
    
    def safe_call(x):
        nonlocal calls
        if time.time() > deadline:
            return None
        if calls >= max_calls:
            return None
        val = f(x)
        calls += 1
        return val

    x0 = float(x0)
    fx0 = safe_call(x0)
    if fx0 is None:
        return x0
    if abs(fx0) <= tol:
        return x0
    
    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) < 0
    # Start with a small step
    h = 1e-4 * max(1.0, abs(x0))
    a = x0
    fa = fx0
    
    # Try expanding to the right first
    b = x0 + h
    fb = safe_call(b)
    if fb is None:
        return a if abs(fa) < tol else a
        
    bracket_found = False
    if fa * fb < 0:
        bracket_found = True
    else:
        # Expand interval to find sign change
        # Alternate direction if right expansion fails for a few steps? 
        # For polynomials, expanding in one direction usually works if x0 is not exactly at a local extremum with no root nearby.
        # We'll try expanding right, then left if needed.
        
        # Try right expansion
        step = h
        found_right = False
        for _ in range(20):
            if time.time() > deadline:
                break
            b = a + step
            fb = safe_call(b)
            if fb is None:
                break
            if fa * fb < 0:
                bracket_found = True
                found_right = True
                break
            step *= 2.0
            # If b is getting huge, stop
            if abs(b) > 1e100:
                break
        
        if not bracket_found:
            # Try left expansion
            step = h
            for _ in range(20):
                if time.time() > deadline:
                    break
                b = a - step
                fb = safe_call(b)
                if fb is None:
                    break
                if fa * fb < 0:
                    bracket_found = True
                    break
                step *= 2.0
                if abs(b) < -1e100:
                    break

    if bracket_found:
        # Ensure a < b for consistency, though sign check works either way
        # We have a, fa, b, fb with fa*fb < 0
        # Phase 2: Hybrid Bisection-Secant
        # We want to minimize calls. Secant is faster.
        # Keep track of last two points for secant if we deviate from bracket endpoints?
        # Standard hybrid:
        # x_s = secant estimate
        # if x_s in (a,b), x_new = x_s else x_new = mid(a,b)
        
        # Initialize last two points for secant calculation
        # We have (a, fa) and (b, fb)
        x_prev1 = a
        f_prev1 = fa
        x_prev2 = b
        f_prev2 = fb
        
        # Main iteration
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Check convergence of current best
            # We track the point with smallest |f|
            # In this loop, we update a or b. The new point is x_new.
            # Let's just track the best point found so far.
            
            # Secant estimate
            if f_prev2 == f_prev1:
                x_new = (a + b) / 2.0
            else:
                x_new = x_prev1 - f_prev1 * (x_prev2 - x_prev1) / (f_prev2 - f_prev1)
            
            # Safeguard: if secant step is outside bracket, use bisection
            if not (a < x_new < b):
                x_new = (a + b) / 2.0
            
            f_new = safe_call(x_new)
            if f_new is None:
                break
            
            if abs(f_new) <= tol:
                return x_new
            
            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
            
            # Update history for secant
            # We need the two most recent points. 
            # If we used secant, the new point is x_new. The previous point was x_prev1 or x_prev2?
            # Actually, standard secant uses the last two points evaluated.
            # Let's maintain a list or just shift.
            # However, if we bisected, the "secant" history is broken in terms of locality, but mathematically it's still two points.
            # To be safe and fast, we can just use the current bracket endpoints for the next secant calculation?
            # No, that's not right. Secant uses the last two *evaluated* points.
            # Let's stick to using the last two evaluated points (x_prev1, f_prev1) and (x_new, f_new).
            # But we also need to maintain the bracket [a, b].
            
            x_prev2 = x_prev1
            f_prev2 = f_prev1
            x_prev1 = x_new
            f_prev1 = f_new
            
            # Check if bracket is small enough
            if abs(b - a) < 1e-12 * max(1.0, abs(a)):
                # Return the point with smaller |f|
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b

        # If loop finishes without hitting tol, return best
        if abs(fa) < abs(fb):
            return a
        else:
            return b

    else:
        # No bracket found. Fallback to minimization / secant on best points
        # This case is rare for polynomials with real roots if x0 is reasonable.
        # We will try to minimize |f(x)| using a simple secant-like approach on the best points found.
        # We have a, fa, and potentially other points from expansion.
        # Let's just use the best point found so far and try to improve it.
        
        best_x = a
        best_f = fa
        
        # Check all points evaluated during expansion? We didn't store them all.
        # Let's just do a local search around best_x.
        # Since we failed to bracket, maybe the root is complex or x0 is far.
        # For the purpose of this task, if we can't bracket, we return the best found.
        # But let's try a few secant steps from the best point and its neighbor.
        
        # We have a and the last b from expansion.
        # Let's try secant between a and the last b evaluated.
        # We need to recover the last b. It's lost in the scope of the failed bracketing loop if not stored.
        # Let's re-evaluate or just return best_x.
        # Given the constraints, returning the best found is safer than guessing.
        
        return best_x

# EVOLVE-BLOCK-END
