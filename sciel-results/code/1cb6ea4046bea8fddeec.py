import math
import time

def find_root(f, x0):
    """
    Finds a root of f near x0.
    Uses a hybrid Newton-Bisection method with a robust bracketing phase.
    """
    start_time = time.time()
    # Safety margin: The total limit is 150s for 3 equations.
    # We allow 40s per equation to be safe against overhead.
    deadline = start_time + 40.0 

    x0 = float(x0)
    
    # Phase 1: Bracketing
    # We want to find an interval [a, b] such that f(a) and f(b) have opposite signs.
    # We sample points around x0 to find the closest sign change.
    
    # Generate candidate offsets
    # Start small, go larger.
    offsets = [0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0, 20.0, -20.0]
    
    points = []
    values = {}
    
    # Evaluate f at candidate points
    for off in offsets:
        if time.time() > deadline:
            break
        x_try = x0 + off
        if x_try == x0:
            continue
        # Avoid duplicates if any
        if x_try in values:
            continue
        try:
            val = f(x_try)
        except Exception:
            continue
        if not math.isfinite(val):
            continue
        points.append(x_try)
        values[x_try] = val
        
    # Include x0 itself
    if x0 not in values:
        try:
            val0 = f(x0)
            if math.isfinite(val0):
                points.append(x0)
                values[x0] = val0
        except Exception:
            pass

    if not points:
        return x0 # Fallback, shouldn't happen

    # Sort points to check adjacent intervals
    points.sort()
    
    a, b = None, None
    fa, fb = None, None
    
    # Find the smallest interval with a sign change
    min_width = float('inf')
    for i in range(len(points) - 1):
        p1 = points[i]
        p2 = points[i+1]
        f1 = values[p1]
        f2 = values[p2]
        
        # Check for sign change
        if f1 * f2 < 0:
            width = abs(p2 - p1)
            if width < min_width:
                min_width = width
                a, b = p1, p2
                fa, fb = f1, f2
        elif f1 == 0:
            return p1
        elif f2 == 0:
            return p2

    # If no bracket found, try to find one by expanding further
    if a is None:
        # Expand search range
        current_range = 20.0
        found = False
        while current_range < 1000.0 and not found:
            if time.time() > deadline:
                break
            x_left = x0 - current_range
            x_right = x0 + current_range
            
            try:
                fl = f(x_left)
                fr = f(x_right)
            except Exception:
                current_range *= 2
                continue
                
            if math.isfinite(fl) and math.isfinite(fr):
                if fl * fr < 0:
                    a, b = x_left, x_right
                    fa, fb = fl, fr
                    found = True
            current_range *= 2
            
        if not found:
            # Return best guess if no bracket found
            # Find point with smallest |f(x)|
            best_x = min(points, key=lambda x: abs(values[x]))
            return best_x

    # Phase 2: Refinement using Newton's Method with Bisection Safeguard
    # Start from the midpoint of the bracket
    x_curr = (a + b) / 2.0
    f_curr = f(x_curr)
    
    # If the midpoint is already good
    if abs(f_curr) < 1e-8:
        return x_curr

    for _ in range(50): # Max Newton iterations
        if time.time() > deadline:
            break
            
        if abs(f_curr) < 1e-12:
            return x_curr

        # Estimate derivative using Central Difference
        # h should be small but not too small to avoid cancellation
        h = 1e-5 * max(1.0, abs(x_curr))
        
        try:
            f_plus = f(x_curr + h)
            f_minus = f(x_curr - h)
        except Exception:
            # Fallback to forward difference or bisection
            f_plus = f(x_curr + h)
            f_minus = f_curr
            
        df = (f_plus - f_minus) / (2 * h)
        
        if df == 0 or not math.isfinite(df):
            # Bisect
            x_next = (a + b) / 2.0
        else:
            x_next = x_curr - f_curr / df
            
        # Check if x_next is within the bracket
        if x_next < a or x_next > b:
            # If not, use bisection step
            x_next = (a + b) / 2.0
            
        # Evaluate f at x_next
        try:
            f_next = f(x_next)
        except Exception:
            # If evaluation fails, bisect
            x_next = (a + b) / 2.0
            f_next = f(x_next)
            
        if not math.isfinite(f_next):
            x_next = (a + b) / 2.0
            f_next = f(x_next)
            
        # Update Bracket
        # The root is in the sub-interval where the sign changes
        if f_curr * f_next < 0:
            # Root is between x_curr and x_next
            a, b = x_curr, x_next
            fa, fb = f_curr, f_next
        else:
            # Root is in the other part of the bracket
            # Determine which side contains the root
            if fa * f_next < 0:
                # Root is between a and x_next
                b = x_next
                fb = f_next
            else:
                # Root is between x_next and b
                a = x_next
                fa = f_next
                
        x_curr, f_curr = x_next, f_next

    return x_curr
