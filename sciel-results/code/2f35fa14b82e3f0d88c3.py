import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses a hybrid approach: attempts to bracket the root first, then uses 
    Brent's method (Bisection + Inverse Quadratic Interpolation) for robustness 
    and speed. Falls back to Secant with safeguards if bracketing fails.
    """
    start_time = time.time()
    deadline = start_time + 10.0  # 10 seconds is plenty for <500 calls
    
    # Tolerance for root
    tol = 1e-12  # Tighter internal tolerance to ensure final check passes
    max_iter = 100
    
    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0

    # Try to find a bracket [a, b] such that f(a) * f(b) < 0
    # Start with a small step and expand
    # Adaptive step size based on x0
    base_step = 1e-4 * max(1.0, abs(x0))
    
    a = x0
    fa = f0
    b = None
    fb = None
    
    # Search for a sign change
    # Try a few steps in both directions
    steps = [1, -1, 2, -2, 4, -4, 8, -8, 16, -16]
    for s in steps:
        x_try = x0 + s * base_step
        f_try = f(x_try)
        
        if not math.isfinite(f_try):
            continue
            
        if abs(f_try) <= tol:
            return x_try
            
        if fa * f_try < 0:
            # Found a bracket
            if abs(fa) < abs(f_try):
                a = x0
                b = x_try
                fa = f0
                fb = f_try
            else:
                a = x_try
                b = x0
                fa = f_try
                fb = f0
            break
    
    if b is not None:
        # Use Brent's Method (simplified: Bisection + Inverse Quadratic Interpolation)
        # We maintain a bracket [a, b] with f(a)*f(b) < 0
        # c is the best estimate
        c = b
        fc = fb
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            # If f(a) is small enough, return a
            if abs(fa) <= tol:
                return a
            # If f(c) is small enough, return c
            if abs(fc) <= tol:
                return c
            
            # Check if we need to bisect
            # If the interval is small, bisect
            if abs(b - a) < tol:
                return (a + b) / 2.0
            
            # Try Inverse Quadratic Interpolation
            # Points: (a, fa), (b, fb), (c, fc)
            # We need 3 distinct points. If not, use Secant on (b, c) or (a, c)
            
            # Check if we have 3 distinct points
            if a != b and b != c and a != c:
                # Lagrange interpolation for x where y=0
                # x = (a*b*fc + b*c*fa + c*a*fb) / (b*fc + c*fa + a*fb - a*fb - b*fa - c*fb) ... wait, formula is complex.
                # Standard formula for inverse quadratic interpolation:
                # Let's use the standard Brent's update logic.
                
                # If fa == fb, use secant on (b, c) or similar.
                # Simpler: Use Secant method on the two most recent points that are likely to converge.
                # But Brent's is safer. Let's implement the standard step.
                
                # Calculate the next point x_new
                # We want to interpolate 0 from (a,fa), (b,fb), (c,fc).
                # If fa == fb:
                #   x_new = c - fc * (c - b) / (fc - fb)  # Secant on b, c
                # elif fb == fc:
                #   x_new = c - fc * (c - a) / (fc - fa)  # Secant on a, c
                # else:
                #   x_new = a * (fb*fc) / ((fa-fb)*(fa-fc)) + b * (fa*fc) / ((fb-fa)*(fb-fc)) + c * (fa*fb) / ((fc-fa)*(fc-fb))
                
                if fa == fb:
                    denom = fc - fb
                    if denom != 0:
                        x_new = c - fc * (c - b) / denom
                    else:
                        x_new = (a + b) / 2.0
                elif fb == fc:
                    denom = fc - fa
                    if denom != 0:
                        x_new = c - fc * (c - a) / denom
                    else:
                        x_new = (a + b) / 2.0
                else:
                    term1 = a * (fb * fc) / ((fa - fb) * (fa - fc))
                    term2 = b * (fa * fc) / ((fb - fa) * (fb - fc))
                    term3 = c * (fa * fb) / ((fc - fa) * (fc - fb))
                    x_new = term1 + term2 + term3
            else:
                # Not enough distinct points, use Secant on the two endpoints of the bracket
                # Or just bisect
                denom = fb - fa
                if denom != 0:
                    x_new = b - fb * (b - a) / denom
                else:
                    x_new = (a + b) / 2.0
            
            # Check if x_new is within the bracket
            # If not, or if it's too close to the midpoint (to ensure linear convergence), use Bisection
            mid = (a + b) / 2.0
            if not (a < x_new < b or b < x_new < a):
                x_new = mid
            else:
                # Check if x_new is close to mid
                # If |x_new - mid| >= 0.5 * |b - a|, use mid
                if abs(x_new - mid) >= 0.5 * abs(b - a):
                    x_new = mid
            
            f_new = f(x_new)
            
            if not math.isfinite(f_new):
                # Diverged, bisect
                x_new = mid
                f_new = f(x_new)
                if not math.isfinite(f_new):
                    break
            
            # Update bracket
            if fa * f_new < 0:
                b = x_new
                fb = f_new
            else:
                a = x_new
                fa = f_new
            
            # Update c (best estimate)
            if abs(fa) < abs(fc):
                c = a
                fc = fa
            elif abs(fb) < abs(fc):
                c = b
                fc = fb
            # else c remains, but we should check if x_new is better
            if abs(f_new) < abs(fc):
                c = x_new
                fc = f_new

        return c

    else:
        # No bracket found. Use Secant Method with safeguards.
        # Initialize second point
        x_prev = x0
        f_prev = f0
        
        # Try a few steps to find a better starting point for Secant
        # We want a point where the slope is not zero and we are moving towards a root
        x_curr = x0 + base_step
        f_curr = f(x_curr)
        
        if abs(f_curr) <= tol:
            return x_curr
            
        best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
        best_f = min(abs(f_prev), abs(f_curr))
        
        for _ in range(max_iter):
            if time.time() > deadline:
                break
                
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                # Slope is zero, break
                break
                
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom
            f_next = f(x_next)
            
            if not math.isfinite(f_next) or not math.isfinite(x_next):
                break
                
            if abs(f_next) < best_f:
                best_x = x_next
                best_f = abs(f_next)
                
            if abs(f_next) <= tol:
                return x_next
                
            x_prev = x_curr
            f_prev = f_curr
            x_curr = x_next
            f_curr = f_next
            
        return best_x
