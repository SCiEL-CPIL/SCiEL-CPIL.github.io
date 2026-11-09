"""IQI-based root finder for oscillatory equations with deadline guard."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """Find a root of f starting near x0 using IQI/secant hybrid."""
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill
    
    x0 = float(x0)
    
    # Track best (closest to zero)
    best_x = x0
    best_fx = f(x0)
    best_abs = abs(best_fx)
    
    # If already at root
    if best_abs <= 1e-8:
        return best_x
    
    # We'll maintain a small history of (x, fx) for IQI
    # Start with 2 points for secant
    # Choose h based on scale
    h0 = 1e-4 * max(1.0, abs(x0))
    
    x1 = x0 + h0
    fx1 = f(x1)
    
    if abs(fx1) <= 1e-8:
        return x1
    
    if abs(fx1) < best_abs:
        best_x = x1
        best_fx = fx1
        best_abs = abs(fx1)
    
    # Now we have (x0, fx0) and (x1, fx1)
    # If sign change, we can use secant directly
    # Otherwise, try to get a 3rd point for IQI
    
    x0_save = x0
    fx0_save = best_fx  # use the original x0's fx
    x1_save = x1
    fx1_save = fx1
    
    # Check if we have a sign change between x0 and x1
    # If yes, use secant within that bracket
    # If no, we need to expand to find a sign change or just proceed with IQI
    
    # Strategy: Use IQI with 3 points. We need a 3rd point.
    # Try a point on the other side of x0
    x2 = x0 - h0
    fx2 = f(x2)
    
    if abs(fx2) <= 1e-8:
        return x2
    
    if abs(fx2) < best_abs:
        best_x = x2
        best_fx = fx2
        best_abs = abs(fx2)
    
    # Now we have 3 points: (x0, fx0), (x1, fx1), (x2, fx2)
    # Use IQI to predict next point
    # IQI: fit quadratic x = a*f^2 + b*f + c through the 3 points
    # Then find f=0 => x = c
    
    def iqi_step(xa, fa, xb, fb, xc, fc):
        """Inverse quadratic interpolation step."""
        # Fit x = a*f^2 + b*f + c
        # Using Lagrange interpolation
        if abs(fa - fb) < 1e-300:
            return None
        if abs(fa - fc) < 1e-300:
            return None
        if abs(fb - fc) < 1e-300:
            return None
        
        # x(0) = a*0 + b*0 + c = c
        # So c = x at f=0
        # Using the three points:
        # xa = a*fa^2 + b*fa + c
        # xb = a*fb^2 + b*fb + c
        # xc = a*fc^2 + b*fc + c
        
        # Subtract to eliminate c:
        # xa - xb = a*(fa^2 - fb^2) + b*(fa - fb)
        # xa - xc = a*(fa^2 - fc^2) + b*(fa - fc)
        
        d1 = fa - fb
        d2 = fa - fc
        d3 = fb - fc
        
        if abs(d1) < 1e-300 or abs(d2) < 1e-300 or abs(d3) < 1e-300:
            return None
        
        # Solve for a and b
        # From xa - xb = a*(fa+fb)*d1 + b*d1
        # => (xa - xb)/d1 = a*(fa+fb) + b
        # Similarly (xa - xc)/d2 = a*(fa+fc) + b
        
        s1 = (xa - xb) / d1
        s2 = (xa - xc) / d2
        
        # s1 = a*(fa+fb) + b
        # s2 = a*(fa+fc) + b
        # s1 - s2 = a*(fb - fc) = a*d3
        if abs(d3) < 1e-300:
            return None
        a = (s1 - s2) / d3
        b = s1 - a * (fa + fb)
        c = xa - a * fa * fa - b * fa
        
        x_new = c  # x at f=0
        
        return x_new
    
    # Main iteration loop
    # Maintain last 3 points
    pts = [(x0_save, fx0_save), (x1_save, fx1_save), (x2, fx2)]
    
    max_iter = 100
    
    for iteration in range(max_iter):
        # Check deadline
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if best is good enough
        if best_abs <= 1e-8:
            return best_x
        
        # Try IQI if we have 3 distinct points
        xa, fa = pts[0]
        xb, fb = pts[1]
        xc, fc = pts[2]
        
        x_new = iqi_step(xa, fa, xb, fb, xc, fc)
        
        # If IQI fails or gives a bad point, use secant with the two closest to zero
        if x_new is None or not math.isfinite(x_new):
            # Use secant with the two points that bracket a sign change if possible
            # Otherwise use the two points closest to zero
            # Sort by |f|
            sorted_pts = sorted(pts, key=lambda p: abs(p[1]))
            xa, fa = sorted_pts[0]
            xb, fb = sorted_pts[1]
            
            if abs(fa - fb) < 1e-300:
                # Degenerate, take a small step
                x_new = xa + 1e-6 * max(1.0, abs(xa))
            else:
                x_new = xa - fa * (xb - xa) / (fb - fa)
        
        # Evaluate
        fx_new = f(x_new)
        
        if not math.isfinite(fx_new):
            # Bad point, shrink step
            x_new = 0.5 * (xa + xb)
            fx_new = f(x_new)
            if not math.isfinite(fx_new):
                break
        
        if abs(fx_new) <= 1e-8:
            return x_new
        
        if abs(fx_new) < best_abs:
            best_x = x_new
            best_fx = fx_new
            best_abs = abs(fx_new)
        
        # Update points: keep the 3 most useful points
        # Strategy: keep the 3 points with smallest |f|, but ensure we don't lose bracketing info
        # For oscillatory functions, we want recent points
        pts = [(xa, fa), (xb, fb), (x_new, fx_new)]
        # Actually, let's keep the 3 points that are most recent and diverse
        # Simpler: just use the last 3 points
        # But for IQI, we want points that are well-separated in f-value
        # Let's keep the 3 with largest pairwise |f_i - f_j|
        
        # For now, just use the last 3
        # pts = [(xb, fb), (xc, fc), (x_new, fx_new)]  # shift window
    
    return best_x
# EVOLVE-BLOCK-END
