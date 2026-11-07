import math
import time
import numpy as np
import scipy.optimize

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Check if x0 is a root.
    2. Attempt to identify if f is a low-degree polynomial (deg <= 5) by 
       evaluating at 6 points and checking finite differences.
    3. If polynomial structure is detected, use numpy.roots to find analytic roots.
    4. Select the best root (closest to x0, smallest residual).
    5. If polynomial reconstruction fails or verification fails, fall back to 
       robust hybrid bracketing/Newton method.
    """
    TOL = 1e-8
    DEADLINE = 140.0
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # 1. Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    # 2. Polynomial Reconstruction Attempt
    # We need at least N+1 points for degree N. Let's try up to degree 5.
    # We will evaluate at x0, x0+h, x0+2h, ... x0+5h
    # But we already have f(x0).
    
    # Choose a step size h. 
    # If x0 is 0, h=1. If x0 is large, h should scale.
    # However, for polynomial reconstruction, the points don't strictly need to be "local" 
    # in the sense of convergence, just distinct. But being close helps avoid 
    # numerical instability in the Vandermonde matrix if coefficients are huge.
    # Let's use a small step relative to x0 magnitude, but not too small to avoid 
    # cancellation errors in finite differences.
    
    scale = max(1.0, abs(x0))
    h = 1e-3 * scale
    
    # Points: x0, x0+h, x0+2h, x0+3h, x0+4h, x0+5h
    # We have f(x0). Need 5 more calls.
    # Total calls so far: 1 (x0) + 5 = 6.
    
    x_vals = [x0]
    f_vals = [fx0]
    
    for i in range(1, 6):
        if not check_deadline():
            break
        x_i = x0 + i * h
        try:
            f_i = f(x_i)
        except Exception:
            f_i = None
        
        if f_i is None or not is_finite(f_i):
            break
            
        x_vals.append(x_i)
        f_vals.append(f_i)
        
    # If we didn't get 6 points, we can't reliably determine degree up to 5.
    # Try with whatever we have, but require at least 3 points (degree 2).
    if len(x_vals) < 3:
        # Fall back to robust method
        return _robust_find_root(f, x0, fx0, START_TIME, DEADLINE)

    # Check finite differences to estimate degree
    # Delta^k f is constant for degree k.
    # Let's compute differences for the available points.
    
    def get_differences(vals):
        diffs = [vals]
        while len(diffs[-1]) > 1:
            curr = diffs[-1]
            next_diff = [curr[i+1] - curr[i] for i in range(len(curr)-1)]
            diffs.append(next_diff)
        return diffs

    try:
        diffs = get_differences(f_vals)
    except Exception:
        return _robust_find_root(f, x0, fx0, START_TIME, DEADLINE)
    
    # Determine if it looks like a polynomial
    # For degree d, the d-th difference should be constant (or close to it).
    # The (d+1)-th difference should be ~0.
    
    # We have up to 6 points.
    # If len(x_vals) == 6, we have diffs up to level 5.
    # Let's check levels from highest available down to 1.
    
    # Note: Numerical noise. If the polynomial has large coefficients, 
    # finite differences might be noisy.
    # However, numpy.roots is robust enough if we provide the coefficients 
    # derived from interpolation.
    
    # Instead of strict finite difference check, let's just try to fit a polynomial 
    # of degree len(x_vals)-1 and see if it fits well.
    # Or better: try degrees from low to high.
    
    best_root = None
    best_resid = float('inf')
    
    # Try to fit polynomials of degree 1, 2, 3, 4, 5
    max_deg = len(x_vals) - 1
    
    for deg in range(1, max_deg + 1):
        if len(x_vals) < deg + 1:
            break
            
        xs = x_vals[:deg+1]
        ys = f_vals[:deg+1]
        
        # Use numpy polyfit
        try:
            coeffs = np.polyfit(xs, ys, deg)
        except Exception:
            continue
            
        # Check fit quality: residual at the other points (if any)
        # If we used all points, residual is 0 by construction (interpolation).
        # But if the function isn't a polynomial of that degree, 
        # the "next" point (if available) should have large residual.
        
        # If we used all available points, we assume it's a polynomial of that degree.
        # To be safer, if we have more points than deg+1, check the rest.
        if len(x_vals) > deg + 1:
            # Check residual at x_vals[deg+1]
            x_check = x_vals[deg+1]
            y_check = f_vals[deg+1]
            y_pred = np.polyval(coeffs, x_check)
            resid = abs(y_check - y_pred)
            
            # If residual is large, it's not a polynomial of this degree.
            # Tolerance for "large": relative to magnitude of y.
            scale_y = max(1.0, abs(y_check))
            if resid > 1e-6 * scale_y:
                continue
            # If we passed the check, it's likely a polynomial of this degree.
            # We can try to fit using ALL points for better numerical stability?
            # No, if it's degree deg, fitting with deg+1 points is exact. 
            # Using more points might introduce noise if it's not exactly deg.
            # But if the check passed, it's very likely deg.
            
        # Now find roots
        try:
            roots = np.roots(coeffs)
        except Exception:
            continue
            
        # Filter real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)):
                real_roots.append(r.real)
                
        if not real_roots:
            continue
            
        # Select root closest to x0
        # Also verify with f()
        for r in real_roots:
            if not is_finite(r):
                continue
            try:
                fr = f(r)
            except Exception:
                continue
                
            if not is_finite(fr):
                continue
                
            if abs(fr) <= TOL:
                # We found a valid root
                # Prefer the one closest to x0? 
                # The task says "returns x with |f(x)| <= 1e-8". 
                # It doesn't specify which root, but usually local context implies closest.
                # However, if multiple roots satisfy, any is fine.
                # But for stability, closest to x0 is safer.
                if best_root is None or abs(r - x0) < abs(best_root - x0):
                    best_root = r
                    best_resid = abs(fr)
                    
    if best_root is not None:
        return best_root

    # If polynomial reconstruction didn't yield a verified root, fall back
    return _robust_find_root(f, x0, fx0, START_TIME, DEADLINE)


def _robust_find_root(f, x0, fx0, START_TIME, DEADLINE):
    """
    Fallback robust root finder: Bracketing + Brent, then Safeguarded Newton.
    """
    TOL = 1e-8
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    if abs(fx0) <= TOL:
        return x0

    # Strategy 1: Bracketing + Brent's Method
    a = x0 - 1.0
    b = x0 + 1.0
    fa = None
    fb = None
    
    bracket_found = False
    for i in range(30):
        if not check_deadline():
            break
            
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
            
        if not (is_finite(fa) and is_finite(fb)):
            break
            
        if fa == 0:
            return a
        if fb == 0:
            return b
            
        if fa * fb < 0:
            bracket_found = True
            break
            
        # Expand
        a = x0 - (x0 - a) * 2
        b = x0 + (b - x0) * 2
        
        if abs(a) > 1e10 or abs(b) > 1e10:
            break

    if bracket_found:
        try:
            root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
            if is_finite(root):
                return root
        except Exception:
            pass

    # Strategy 2: Safeguarded Newton's Method
    x = float(x0)
    fx = fx0
    
    for i in range(60):
        if not check_deadline():
            break
            
        if abs(fx) <= TOL:
            return x
            
        h = 1e-6 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except Exception:
            break
            
        if not is_finite(fx_plus):
            break
            
        dfx = (fx_plus - fx) / h
        
        if dfx == 0:
            x_new = x + 1e-4
            try:
                fx = f(x_new)
            except Exception:
                break
            if is_finite(fx):
                x = x_new
            else:
                break
            continue
            
        x_new = x - fx / dfx
        
        if not is_finite(x_new) or abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            x_new = x - 0.5 * (fx / dfx)
            
        if not is_finite(x_new):
            break
            
        try:
            fx_new = f(x_new)
        except Exception:
            break
            
        if not is_finite(fx_new):
            x_new = x - 0.1 * (fx / dfx)
            if not is_finite(x_new):
                break
            try:
                fx_new = f(x_new)
            except Exception:
                break
            if not is_finite(fx_new):
                break
                
        x = x_new
        fx = fx_new
        
    return x
# EVOLVE-BLOCK-END
