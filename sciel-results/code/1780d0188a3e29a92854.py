"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0.
    Strategy: Hybrid Steffensen's Method with Adaptive Step and Bisection Safety.
    
    1. Initial exploration to find a bracket [a, b] where f(a) and f(b) have opposite signs.
       If no bracket is found, we use a global search for the minimum |f(x)|.
    2. Once a bracket is found, we use Steffensen's method (fixed-point iteration with 
       2-point derivative approximation) for quadratic convergence.
    3. If Steffensen's step leaves the bracket or diverges, we fall back to Bisection 
       to guarantee linear convergence.
    4. We track the best |f(x)| seen to ensure we return a valid root if possible.
    """
    deadline = time.time() + 145.0
    tol = 1e-9
    x0 = float(x0)
    
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f) if math.isfinite(best_f) else float('inf')
    
    if best_abs <= tol:
        return best_x

    def update_best(x_val, f_val):
        nonlocal best_x, best_f, best_abs
        if not math.isfinite(x_val) or f_val is None or not math.isfinite(f_val):
            return
        abs_f = abs(f_val)
        if abs_f < best_abs:
            best_x = x_val
            best_f = f_val
            best_abs = abs_f

    # --- Phase 1: Bracketing / Global Search ---
    # We need to find a bracket [a, b] such that f(a)*f(b) < 0.
    # For oscillatory functions, we scan outwards from x0.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # We also track the best point found during the scan in case no bracket is found
    scan_best_x = x0
    scan_best_f = best_f
    scan_best_abs = best_abs

    # Scales for scanning
    # We want to cover a wide range quickly.
    scales = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4]
    
    # We will check pairs (x0 - s, x0) and (x0, x0 + s)
    # And also (x0 - s, x0 + s) to catch roots between them
    
    for s in scales:
        if time.time() > deadline:
            break
            
        # Check x0 - s
        x_left = x0 - s
        try:
            f_left = f(x_left)
        except Exception:
            f_left = float('inf')
        
        update_best(x_left, f_left)
        if math.isfinite(f_left):
            if abs(f_left) < scan_best_abs:
                scan_best_x = x_left
                scan_best_f = f_left
                scan_best_abs = abs(f_left)
                
        # Check x0 + s
        x_right = x0 + s
        try:
            f_right = f(x_right)
        except Exception:
            f_right = float('inf')
            
        update_best(x_right, f_right)
        if math.isfinite(f_right):
            if abs(f_right) < scan_best_abs:
                scan_best_x = x_right
                scan_best_f = f_right
                scan_best_abs = abs(f_right)
                
        if best_abs <= tol:
            return best_x

        # Check for bracket [x0 - s, x0]
        if math.isfinite(f_left) and math.isfinite(best_f):
            if f_left * best_f < 0:
                a, b = x_left, x0
                fa, fb = f_left, best_f
                break
        
        # Check for bracket [x0, x0 + s]
        if math.isfinite(best_f) and math.isfinite(f_right):
            if best_f * f_right < 0:
                a, b = x0, x_right
                fa, fb = best_f, f_right
                break
                
        # Check for bracket [x0 - s, x0 + s]
        if math.isfinite(f_left) and math.isfinite(f_right):
            if f_left * f_right < 0:
                a, b = x_left, x_right
                fa, fb = f_left, f_right
                break

    # --- Phase 2: Root Finding ---
    
    if a is not None and b is not None:
        # We have a bracket [a, b]
        # Use Hybrid Steffensen/Bisection
        # Steffensen's method: x_new = x - f(x) / (f(x+h) - f(x))/h
        # Here we use the bracket to ensure stability.
        
        # Start with the midpoint or the point closer to 0? 
        # Let's start with the point that has smaller |f|
        if abs(fa) < abs(fb):
            x_curr = a
            f_curr = fa
        else:
            x_curr = b
            f_curr = fb
            
        # Ensure x_curr is inside [a, b]
        # If x_curr is a, then f_curr = fa. If b, f_curr = fb.
        
        # Adaptive h for Steffensen
        # h should be small enough to approximate derivative, large enough to avoid noise.
        # For oscillatory functions, h should be related to the local curvature.
        # Let's start with h = (b - a) * 0.1 and adjust.
        
        max_iters = 50
        for _ in range(max_iters):
            if time.time() > deadline:
                break
                
            if abs(f_curr) <= tol:
                return x_curr
                
            # Calculate h
            # If we are close to the boundary, reduce h
            # h = 1e-4 * (b - a) is a reasonable start
            h = 1e-4 * (b - a)
            if h == 0:
                h = 1e-10
                
            # Ensure x_curr + h is within [a, b] if possible, or at least finite
            x_test = x_curr + h
            # If x_test is outside, try x_curr - h
            if x_test > b or x_test < a:
                x_test = x_curr - h
            if x_test > b or x_test < a:
                # If still outside, just use a tiny h
                x_test = x_curr + 1e-10
                
            try:
                f_test = f(x_test)
            except Exception:
                f_test = float('inf')
                
            update_best(x_test, f_test)
            
            if not math.isfinite(f_test):
                # Fallback to bisection
                x_mid = (a + b) / 2.0
                f_mid = f(x_mid)
                update_best(x_mid, f_mid)
                
                if f_curr * f_mid <= 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                    
                x_curr = x_mid
                f_curr = f_mid
                continue
            
            # Steffensen step
            # Derivative approx: (f_test - f_curr) / (x_test - x_curr)
            denom = f_test - f_curr
            if denom == 0 or not math.isfinite(denom):
                # Fallback to bisection
                x_mid = (a + b) / 2.0
                f_mid = f(x_mid)
                update_best(x_mid, f_mid)
                
                if f_curr * f_mid <= 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                    
                x_curr = x_mid
                f_curr = f_mid
                continue
            
            # dx = f_curr / derivative
            # derivative = denom / (x_test - x_curr)
            # dx = f_curr * (x_test - x_curr) / denom
            dx = f_curr * (x_test - x_curr) / denom
            
            x_new = x_curr - dx
            
            # Safety: If x_new is outside [a, b], we might have diverged.
            # For oscillatory functions, a jump outside the bracket might be correct 
            # if the bracket was "wide" and we jumped over a root, but usually 
            # it's safer to bisect if we leave the bracket.
            if x_new < a or x_new > b:
                # Fallback to bisection
                x_mid = (a + b) / 2.0
                f_mid = f(x_mid)
                update_best(x_mid, f_mid)
                
                if f_curr * f_mid <= 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                    
                x_curr = x_mid
                f_curr = f_mid
                continue
            
            # Update bracket
            try:
                f_new = f(x_new)
            except Exception:
                f_new = float('inf')
                
            update_best(x_new, f_new)
            
            if not math.isfinite(f_new):
                # Fallback to bisection
                x_mid = (a + b) / 2.0
                f_mid = f(x_mid)
                update_best(x_mid, f_mid)
                
                if f_curr * f_mid <= 0:
                    b = x_mid
                    fb = f_mid
                else:
                    a = x_mid
                    fa = f_mid
                    
                x_curr = x_mid
                f_curr = f_mid
                continue
            
            if f_curr * f_new <= 0:
                # Root is between x_curr and x_new
                if abs(f_new) < abs(f_curr):
                    a = x_curr
                    fa = f_curr
                    b = x_new
                    fb = f_new
                else:
                    a = x_new
                    fa = f_new
                    b = x_curr
                    fb = f_curr
            else:
                # Same sign, shrink bracket
                if f_curr * fa > 0:
                    a = x_curr
                    fa = f_curr
                else:
                    b = x_curr
                    fb = f_curr
                    
            x_curr = x_new
            f_curr = f_new
            
            if best_abs <= tol:
                return best_x

    # If no bracket was found, or we ran out of iterations, return the best point found
    # We might try a few more local steps from the best point found during scan
    if scan_best_abs < best_abs:
        best_x = scan_best_x
        best_f = scan_best_f
        best_abs = scan_best_abs

    return best_x
# EVOLVE-BLOCK-END
