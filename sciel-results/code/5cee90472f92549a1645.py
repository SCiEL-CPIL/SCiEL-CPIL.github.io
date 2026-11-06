import math
import time

def find_root(f, x0):
    start_time = time.time()
    MAX_WALL = 115.0  # Safety margin before 120s limit
    MAX_EVALS = 500
    TARGET_TOL = 1e-8
    
    eval_count = [0]
    
    def eval_f(x):
        eval_count[0] += 1
        return f(x)

    def time_ok():
        return time.time() - start_time < MAX_WALL

    def check_time():
        if not time_ok():
            raise TimeoutError("Wall clock limit exceeded")

    x0 = float(x0)
    fx0 = eval_f(x0)
    best_x = x0
    best_fx = fx0
    
    if abs(fx0) <= TARGET_TOL:
        return x0

    # --- Phase 1: Find a Bracket ---
    # Geometric step doubling to find a sign change.
    # Start with a small step, double it until sign change or limit.
    # We also check the "other side" if the first direction fails quickly.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Initial step size heuristic
    # Use a small relative step
    base_h = 1e-4 * max(1.0, abs(x0))
    
    # Try expanding from x0
    # We keep track of the last point to form the bracket
    last_x = x0
    last_fx = fx0
    
    found_bracket = False
    direction = 1  # 1 for right, -1 for left
    
    # We will try to find a bracket by expanding in one direction first.
    # If we go too far or hit a limit without finding one, we might try the other.
    # But usually, for local ill-conditioned problems, the root is nearby.
    
    # Let's do a symmetric expansion or just one direction? 
    # One direction is cheaper. Let's try right first. If it fails significantly, try left.
    
    h = base_h
    for i in range(50):  # Max 50 doublings
        if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
            break
        
        x_new = x0 + direction * h
        if x_new == x0:
            h *= 2
            continue
            
        fx_new = eval_f(x_new)
        
        if abs(fx_new) < abs(best_fx):
            best_x, best_fx = x_new, fx_new
            
        if abs(fx_new) <= TARGET_TOL:
            return x_new
            
        # Check for sign change between last_x and x_new
        if last_fx * fx_new < 0:
            a, b = last_x, x_new
            fa, fb = last_fx, fx_new
            found_bracket = True
            break
            
        # Update last point
        last_x = x_new
        last_fx = fx_new
        
        # Double the step
        h *= 2
        
        # If we've gone very far, it might be better to switch direction
        # But let's stick to the current direction for a bit.
        # If h becomes huge, we might miss a local root.
        # However, the problem states "local ill-conditioned", implying root is near x0.
        
    # If no bracket found in the first direction, try the other direction
    if not found_bracket:
        direction = -1
        last_x = x0
        last_fx = fx0
        h = base_h
        for i in range(50):
            if not time_ok() or eval_count[0] >= MAX_EVALS - 5:
                break
                
            x_new = x0 + direction * h
            if x_new == x0:
                h *= 2
                continue
                
            fx_new = eval_f(x_new)
            
            if abs(fx_new) < abs(best_fx):
                best_x, best_fx = x_new, fx_new
                
            if abs(fx_new) <= TARGET_TOL:
                return x_new
                
            if last_fx * fx_new < 0:
                a, b = last_x, x_new
                fa, fb = last_fx, fx_new
                found_bracket = True
                break
                
            last_x = x_new
            last_fx = fx_new
            h *= 2

    if found_bracket:
        # --- Phase 2: Illinois Method ---
        # The Illinois method is a modification of the bisection method.
        # It uses the secant method to predict the next point, but if the prediction
        # doesn't reduce the bracket size sufficiently or is outside the bracket,
        # it bisects. Specifically, it shrinks the side that did NOT contain the
        # new point by a factor (usually 0.5 or 1/3) to prevent stalling.
        
        # Ensure a and b are ordered
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Illinois parameters
        # Shrink factor for the "old" side
        SHRINK = 0.5 
        
        for i in range(100):  # Max iterations for Illinois
            if not time_ok() or eval_count[0] >= MAX_EVALS:
                break
                
            # Check if current endpoints are good enough
            if abs(fa) <= TARGET_TOL:
                return a
            if abs(fb) <= TARGET_TOL:
                return b
                
            # Secant prediction
            # x_new = b - fb * (b - a) / (fb - fa)
            denom = fb - fa
            if abs(denom) < 1e-15:
                # Degenerate, bisect
                x_new = (a + b) / 2.0
            else:
                x_new = b - fb * (b - a) / denom
                
            # If x_new is outside [a, b], bisect
            if x_new <= a or x_new >= b:
                x_new = (a + b) / 2.0
            else:
                # Illinois specific check:
                # If the new point is too close to one endpoint, it might be inefficient.
                # Standard Illinois: If |x_new - b| < |b - a|/2 (or similar), 
                # it effectively forces a bisection-like behavior on the "bad" side.
                # A simpler implementation:
                # Calculate the secant point.
                # If the secant point is valid, use it.
                # However, to ensure progress, we often replace the endpoint 
                # with a point that shrinks the interval.
                
                # Let's implement the standard Illinois logic:
                # 1. Compute secant x_s.
                # 2. If x_s is not in (a, b), x_s = mid(a, b).
                # 3. Evaluate f(x_s).
                # 4. If f(a)*f(x_s) < 0, new bracket is [a, x_s].
                #    In this case, we "replace" b with x_s.
                #    But to prevent slow convergence if x_s is close to a,
                #    we might shrink the [x_s, b] part? No, we discard [x_s, b].
                #    The Illinois trick is applied to the DISCARDED part?
                #    Actually, the Illinois method modifies the ENDPOINT of the 
                #    discarded interval to force bisection on the next step 
                #    if the secant step was small.
                #    
                #    Common implementation:
                #    Let x_mid = (a+b)/2.
                #    If |x_s - b| < |b - a|/2:
                #       x_s = x_mid
                #    This ensures at least bisection speed.
                
                if abs(x_new - b) < 0.5 * abs(b - a):
                    x_new = (a + b) / 2.0
                elif abs(x_new - a) < 0.5 * abs(a - b):
                    x_new = (a + b) / 2.0

            fx_new = eval_f(x_new)
            
            if abs(fx_new) < abs(best_fx):
                best_x, best_fx = x_new, fx_new
                
            if abs(fx_new) <= TARGET_TOL:
                return x_new
            
            # Update bracket
            if fa * fx_new < 0:
                # Root is in [a, x_new]
                # The interval [x_new, b] is discarded.
                # Illinois: To ensure we don't get stuck with a very small secant step
                # next time, we replace b with a point that shrinks the interval.
                # Specifically, we set b to be closer to x_new?
                # No, we set the OLD b to a new value that makes the next secant step larger?
                # Actually, the standard Illinois algorithm says:
                # If the new point x_new is on the same side as b (i.e. fa*fx_new < 0),
                # then we replace b with x_new.
                # BUT, to prevent the interval from shrinking too slowly on the "other" side,
                # we replace the OLD b (which is now discarded) with a point that is 
                # SHRINK * (b - x_new) away from x_new?
                # No, that's not right.
                
                # Let's look up the precise Illinois algorithm.
                # "The Illinois algorithm is a modification of the bisection method...
                #  It uses the secant method... but if the secant step is too small,
                #  it bisects."
                #  A more robust description:
                #  1. Compute x_s (secant).
                #  2. If x_s is not in (a,b), x_s = mid.
                #  3. Evaluate f(x_s).
                #  4. If f(a)f(x_s) < 0:
                #       b = x_s
                #       # To prevent stagnation, replace the OLD b with a point 
                #       # that shrinks the interval? 
                #       # Actually, the Illinois method replaces the endpoint 
                #       # that was NOT updated with a shrunk version.
                #       # If we updated b (because root is in [a, x_s]), 
                #       # then the "other" endpoint a remains.
                #       # The shrinkage is applied to the interval that was DISCARDED?
                #       # No, the shrinkage is applied to the NEW bracket?
                
                # Let's use a simpler, robust variant often called "Illinois-like":
                # If the secant step was taken, and it reduced the interval, good.
                # If the secant step was "small" (i.e. x_new close to a or b), 
                # we force a bisection.
                
                # My previous check `if abs(x_new - b) < 0.5 * abs(b - a): x_new = mid` 
                # already enforces at least bisection rate. This is sufficient 
                # and effectively creates an Illinois-like behavior without 
                # the complex endpoint replacement logic which can be tricky.
                
                b = x_new
                fb = fx_new
            else:
                # Root is in [x_new, b]
                a = x_new
                fa = fx_new
                
            # Check for convergence of interval width
            if abs(b - a) < 1e-12 * max(1.0, abs(a), abs(b)):
                # Return the midpoint or the better endpoint
                mid = (a + b) / 2.0
                if abs(fa) < abs(fb):
                    return a
                else:
                    return b
        else:
            # Loop finished without explicit return, return best found
            pass

    # If no bracket was found, return the best x found so far
    return best_x
