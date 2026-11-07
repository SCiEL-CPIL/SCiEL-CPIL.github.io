import numpy as np
from scipy.optimize import brentq

def find_root(f, x0):
    x0 = float(x0)
    
    # Check if x0 is already a root
    try:
        fx0 = f(x0)
    except:
        fx0 = np.nan
    
    if np.isfinite(fx0) and abs(fx0) <= 1e-8:
        return x0

    # Phase 1: Newton's Method (limited iterations)
    # This is fast if we are close to a root.
    x = x0
    for _ in range(10):
        try:
            fx = f(x)
        except:
            break
            
        if not np.isfinite(fx):
            break
            
        if abs(fx) <= 1e-8:
            return x
            
        # Finite difference derivative
        h = 1e-5 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except:
            break
            
        if not np.isfinite(fx_plus):
            break
            
        d = (fx_plus - fx) / h
        if d == 0 or not np.isfinite(d):
            break
            
        x_new = x - fx / d
        
        # Prevent divergence
        if not np.isfinite(x_new):
            break
            
        # If the step is too large, stop Newton and switch to bracketing
        if abs(x_new - x) > 10.0 * max(1.0, abs(x)):
            break
            
        x = x_new

    # Check if Newton converged
    try:
        fx = f(x)
        if np.isfinite(fx) and abs(fx) <= 1e-8:
            return x
    except:
        pass

    # Phase 2: Bracketing Search
    # Scan for a sign change in a local window around x0
    # Then use brentq to find the root precisely.
    
    # Start with a small window and expand if necessary
    # For oscillatory functions, roots are likely close.
    # We scan from x0 - span to x0 + span
    
    # Strategy:
    # 1. Try a fine scan around x0.
    # 2. If no sign change, try a coarser scan over a larger range.
    
    def scan_for_bracket(center, span, num_points):
        """
        Scans [center - span, center + span] with num_points steps.
        Returns (a, b) if a sign change is found, else None.
        """
        if num_points < 2:
            return None
        
        xs = np.linspace(center - span, center + span, num_points)
        
        prev_x = None
        prev_fx = None
        
        for xi in xs:
            try:
                fxi = f(xi)
            except:
                fxi = np.nan
            
            if not np.isfinite(fxi):
                # Skip non-finite values, but reset previous if needed?
                # Actually, if we hit a non-finite, we can't use it for bracketing.
                # We just continue to next point.
                prev_x = xi
                prev_fx = fxi
                continue
            
            if prev_fx is not None and np.isfinite(prev_fx):
                if prev_fx == 0:
                    return (prev_x, prev_x)
                if fxi == 0:
                    return (xi, xi)
                if prev_fx * fxi < 0:
                    return (prev_x, xi)
            
            prev_x = xi
            prev_fx = fxi
            
        return None

    # Try increasing spans and resolutions
    # Start with a reasonable assumption: root is within +/- 10, step 0.1 (200 points)
    # This costs 200 calls. If found, brentq adds ~15. Total ~215.
    # If not found, expand.
    
    # To save calls, we can start with a coarser scan?
    # No, oscillatory functions might have narrow sign changes.
    # But 200 calls is safe.
    
    # Let's try a few configurations.
    configs = [
        (1.0, 50),   # +/- 1, 50 points (step 0.04). Cost ~50.
        (5.0, 100),  # +/- 5, 100 points (step 0.1). Cost ~100.
        (20.0, 200), # +/- 20, 200 points (step 0.2). Cost ~200.
        (100.0, 400) # +/- 100, 400 points (step 0.5). Cost ~400.
    ]
    
    for span, n_pts in configs:
        bracket = scan_for_bracket(x0, span, n_pts)
        if bracket:
            a, b = bracket
            try:
                # brentq requires f(a) and f(b) to have opposite signs
                # We already checked this in scan_for_bracket, but let's be safe.
                fa = f(a)
                fb = f(b)
                if fa * fb > 0:
                    # Might have been a zero at endpoint or numerical issue
                    # If one is zero, return it.
                    if abs(fa) <= 1e-8: return a
                    if abs(fb) <= 1e-8: return b
                    # Otherwise, try to refine or continue scanning
                    continue
                root = brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                return root
            except Exception:
                continue
    
    # If all else fails, return the best x found so far
    # This will likely fail the |f(x)| <= 1e-8 check, but it's better than None.
    return x0
