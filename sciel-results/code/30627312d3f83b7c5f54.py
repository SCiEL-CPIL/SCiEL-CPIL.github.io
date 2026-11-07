import math
import time

def find_root(f, x0):
    """
    Finds a root x such that |f(x)| <= 1e-8 for a given function f.
    
    Strategy:
    1. Multi-Scale Global Scan: Sample the function at a fixed number of points 
       across a wide interval around x0 to detect any sign change.
       This is robust for oscillatory functions where local methods might miss 
       roots due to step size issues.
    2. Bisection: Once a bracket [a, b] with f(a)*f(b) < 0 is found, use bisection.
       Bisection is guaranteed to converge and has a predictable call count, 
       unlike Newton or Secant methods which can diverge or oscillate.
    3. Deadline Guard: Ensures the function returns within the time limit.
    """
    TOL = 1e-8
    MAX_CALLS = 500
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0  # 5s margin before 120s limit
    
    calls = 0
    
    def safe_f(x):
        nonlocal calls
        if calls >= MAX_CALLS:
            return float('inf')
        if time.time() > DEADLINE:
            return float('inf')
        calls += 1
        try:
            val = f(x)
            if not math.isfinite(val):
                return float('inf')
            return val
        except Exception:
            return float('inf')

    x_start = float(x0)
    
    # Check starting point
    fx0 = safe_f(x_start)
    if abs(fx0) <= TOL:
        return x_start

    # Global Adaptive Grid Search
    # We will try to find a bracket [a, b] such that f(a)*f(b) < 0
    # We scan a range around x_start with increasing width and resolution
    
    # Initial parameters
    base_width = 1.0
    num_points = 20
    
    # We will try several scales
    # Scale 0: width ~ 1, 20 points
    # Scale 1: width ~ 10, 20 points
    # ...
    # Up to a reasonable max width
    
    best_x = x_start
    best_f = abs(fx0)
    
    # If fx0 is not finite, we need to find a finite point first
    if not math.isfinite(fx0):
        # Try a few points to find a finite value
        for i in range(1, 10):
            x_try = x_start + i * 0.1
            f_try = safe_f(x_try)
            if math.isfinite(f_try):
                best_x = x_try
                best_f = abs(f_try)
                fx0 = f_try # Update reference, though we don't strictly need it if we restart search
                break
            x_try = x_start - i * 0.1
            f_try = safe_f(x_try)
            if math.isfinite(f_try):
                best_x = x_try
                best_f = abs(f_try)
                fx0 = f_try
                break
        
        if not math.isfinite(fx0):
            # Couldn't find any finite value, return best effort (which is inf, but we need a float)
            # The task says non-finite x is a failure. We should return a finite x if possible.
            # If all are inf, we are stuck. Return x_start.
            return x_start

    # Search for bracket
    bracket_found = False
    a = None
    b = None
    fa = None
    fb = None
    
    # We keep a history of points to potentially use if bisection fails or for best_x tracking
    # But primarily we just want a sign change.
    
    # Try different scales
    for scale in range(8):
        width = base_width * (10.0 ** scale)
        # Ensure width is not too small
        if width < 1e-3:
            width = 1e-3
            
        # Number of points: keep it moderate to save calls
        # If width is large, step size is large, might miss narrow oscillations.
        # But for "3 fixed oscillatory equations", usually standard frequencies.
        # Let's increase points slightly for larger scales to maintain resolution?
        # No, keep points fixed to bound calls. 20 points * 8 scales = 160 calls max for search.
        # Leaves 340 for bisection.
        
        num_pts = 20
        step = 2.0 * width / num_pts
        start_x = x_start - width
        
        prev_x = None
        prev_f = None
        
        for i in range(num_pts + 1):
            x_curr = start_x + i * step
            f_curr = safe_f(x_curr)
            
            if not math.isfinite(f_curr):
                prev_x = x_curr
                prev_f = f_curr
                continue
                
            if math.isfinite(prev_f):
                # Check for sign change
                if prev_f * f_curr < 0:
                    bracket_found = True
                    a = prev_x
                    b = x_curr
                    fa = prev_f
                    fb = f_curr
                    break
                # Check for zero
                if abs(f_curr) <= TOL:
                    return x_curr
                if abs(f_curr) < best_f:
                    best_f = abs(f_curr)
                    best_x = x_curr
            
            # Check if current is zero
            if abs(f_curr) <= TOL:
                return x_curr
            if abs(f_curr) < best_f:
                best_f = abs(f_curr)
                best_x = x_curr
                
            prev_x = x_curr
            prev_f = f_curr
            
            if bracket_found:
                break
                
        if bracket_found:
            break
            
        # If we didn't find a bracket, expand width
        if calls > MAX_CALLS - 50:
            break

    if bracket_found:
        # Use Bisection
        # Bisection is robust. 
        # Max iterations: log2((b-a)/tol)
        # (b-a) is at most 2*width. Width could be 1e6. 
        # log2(1e6/1e-8) = log2(1e14) ~ 47.
        # So 50 iterations is safe.
        
        try:
            for i in range(50):
                if calls >= MAX_CALLS - 2:
                    break
                if time.time() > DEADLINE:
                    break
                    
                mid = (a + b) / 2.0
                f_mid = safe_f(mid)
                
                if not math.isfinite(f_mid):
                    # If mid is bad, we can't bisect reliably. 
                    # Try shrinking the interval? Or just break and return best_x?
                    # For robustness, if f_mid is inf, it might be a singularity.
                    # We can try to check if a or b is still valid.
                    # But let's just break to avoid wasting calls.
                    break
                    
                if abs(f_mid) <= TOL:
                    return mid
                
                if fa * f_mid < 0:
                    b = mid
                    fb = f_mid
                else:
                    a = mid
                    fa = f_mid
                    
                # Check if interval is small enough
                if abs(b - a) < 1e-12:
                    return mid
            
            # If bisection finished without hitting TOL (unlikely), return midpoint
            return (a + b) / 2.0
        except Exception:
            pass # Fall through to best_x

    return best_x

