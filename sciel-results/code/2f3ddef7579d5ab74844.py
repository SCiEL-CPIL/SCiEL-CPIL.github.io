# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Strategy: Assume f is a polynomial of moderate degree.
    1. Sample f at N distinct points to set up a linear system for coefficients.
    2. Solve for coefficients.
    3. Find all real roots of the reconstructed polynomial.
    4. Select the root closest to x0.
    5. Verify |f(root)| <= 1e-8. If not, fallback to a robust hybrid search.
    """
    DEADLINE = 110.0
    START_TIME = time.time()
    
    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            fx = f(x)
            if is_finite(fx):
                return fx
        except:
            pass
        return None

    x0 = float(x0)
    
    # Check if x0 is already a root
    fx0 = safe_f(x0)
    if fx0 is not None and abs(fx0) <= 1e-8:
        return x0

    # Step 1: Polynomial Reconstruction
    # We assume the polynomial degree is at most D.
    # We try degrees 4, 6, 8, 10, 12 to cover a reasonable range.
    # For each degree D, we need D+1 points.
    # We choose points spread around x0 to capture local behavior but also global shape.
    
    # To avoid numerical instability in Vandermonde systems, we use Chebyshev nodes
    # mapped to an interval around x0. However, the root might be far.
    # Let's try a few interval scales.
    
    best_candidate = x0
    best_error = abs(fx0) if fx0 is not None else float('inf')
    
    # If fx0 was None, we don't have a baseline error, but we must find a root.
    
    # Strategy:
    # 1. Try to fit a polynomial of degree D using points in [x0 - R, x0 + R].
    # 2. Increase R if the fit is poor or no valid root is found.
    
    max_degree = 12
    # We will test degrees 4, 6, 8, 10, 12
    # For each degree, we use D+1 points.
    # To save calls, we can reuse points if we increase degree, but let's keep it simple first.
    # Total calls per degree attempt: D+1.
    # Max calls if we try all: sum(D+1 for D in [4,6,8,10,12]) = 5+7+9+11+13 = 45 calls.
    # This is well within the 500 limit.
    
    # Let's define a helper to fit and find roots
    def try_polynomial_fit(degree, radius):
        if not check_deadline():
            return None, float('inf')
            
        # Generate Chebyshev nodes in [-1, 1]
        # Map to [x0 - radius, x0 + radius]
        # x = x0 + radius * t
        # t = cos((2k+1)*pi / (2*(degree+1))) for k=0..degree
        # Actually, standard Chebyshev nodes for interpolation:
        # t_k = cos(pi * (2k+1) / (2*(degree+1)))
        
        n_pts = degree + 1
        t_vals = np.cos(np.pi * (2 * np.arange(n_pts) + 1) / (2 * n_pts))
        x_vals = x0 + radius * t_vals
        
        # Evaluate f at these points
        y_vals = []
        for x in x_vals:
            if not check_deadline():
                return None, float('inf')
            y = safe_f(x)
            if y is None:
                return None, float('inf')
            y_vals.append(y)
        
        y_arr = np.array(y_vals)
        
        # Check for NaN/Inf in y
        if not np.all(np.isfinite(y_arr)):
            return None, float('inf')
            
        # Solve for coefficients c_0 ... c_degree
        # f(x) = c_0 + c_1 x + ... + c_degree x^degree
        # We can use np.polyfit or solve Vandermonde.
        # np.polyfit is robust. It returns coefficients from highest degree to lowest.
        try:
            coeffs = np.polyfit(x_vals, y_arr, degree)
        except Exception:
            return None, float('inf')
            
        if not np.all(np.isfinite(coeffs)):
            return None, float('inf')
            
        # Find roots of the polynomial
        try:
            roots = np.roots(coeffs)
        except Exception:
            return None, float('inf')
            
        # Filter real roots
        real_roots = []
        for r in roots:
            if abs(r.imag) < 1e-8 * max(1.0, abs(r.real)):
                real_roots.append(r.real)
                
        if not real_roots:
            return None, float('inf')
            
        # Select the real root closest to x0
        # But we must verify it's a root of the ORIGINAL f
        best_x = None
        best_err = float('inf')
        
        for rx in real_roots:
            if not is_finite(rx):
                continue
            # Check distance to x0 to prefer local roots?
            # The task asks for A root, but x0 is a starting point.
            # Usually, "starting point" implies we want the root near x0.
            # Let's prioritize roots close to x0.
            dist = abs(rx - x0)
            
            # Verify with original f
            frx = safe_f(rx)
            if frx is None:
                continue
                
            err = abs(frx)
            
            # We want min err, but if multiple roots have err < 1e-8, pick closest to x0
            if err < best_err:
                best_err = err
                best_x = rx
            elif abs(err - best_err) < 1e-12:
                # Tie in error, pick closer to x0
                if dist < abs(best_x - x0):
                    best_x = rx
                    best_err = err
                    
        if best_x is not None:
            return best_x, best_err
        else:
            return None, float('inf')

    # Try different degrees and radii
    # Radii: 1.0, 10.0, 100.0, 1000.0
    # Degrees: 4, 6, 8, 10, 12
    
    # We want to minimize calls. 
    # If a low-degree fit works, we stop.
    # If not, we try higher degrees or larger radii.
    
    radii = [1.0, 10.0, 100.0, 1000.0, 10000.0]
    degrees = [4, 6, 8, 10, 12]
    
    # Order of attempts:
    # 1. Low degree, small radius (likely local behavior)
    # 2. High degree, small radius (if function is complex locally)
    # 3. Low degree, large radius (if root is far)
    # ...
    
    # Let's iterate through combinations
    # To be efficient, we can stop as soon as we find a valid root (err <= 1e-8)
    # But we should probably check a few to ensure we didn't pick a spurious root from a bad fit?
    # The verification step (safe_f(rx)) ensures it's a root of f.
    # So the first valid root found is a success.
    
    # However, if the first valid root is far from x0, and there's one closer, 
    # does it matter? The task says "returns x with |f(x)| <= 1e-8".
    # It doesn't explicitly say "closest to x0", but x0 is the starting point.
    # In root finding, finding the root near x0 is standard.
    # If we find ANY root, we pass the |f(x)| check.
    # So we can return the first valid root we find to save calls?
    # No, if we return a root that is valid, we are done.
    # The score depends on number of calls.
    # So we should stop at the first success.
    
    # But what if the first "success" is actually a numerical artifact of the fit?
    # The verification `safe_f(rx)` prevents this. If `safe_f(rx)` is small, it IS a root.
    
    # So: Loop through (degree, radius) pairs.
    # Try to fit. Find roots. Verify. If any verified, return it.
    
    # Which order?
    # Start with moderate degree (6 or 8) and moderate radius (1.0 or 10.0).
    # Polynomials in these tasks are often cubic or quartic.
    # Degree 4 is a safe bet.
    
    search_order = [
        (4, 1.0),
        (6, 1.0),
        (8, 1.0),
        (4, 10.0),
        (6, 10.0),
        (8, 10.0),
        (10, 10.0),
        (4, 100.0),
        (6, 100.0),
        (8, 100.0),
        (12, 100.0),
        (4, 1000.0),
        (8, 1000.0),
        (12, 1000.0),
    ]
    
    for deg, rad in search_order:
        if not check_deadline():
            break
            
        root, err = try_polynomial_fit(deg, rad)
        
        if root is not None and err <= 1e-8:
            return root
            
        # If we found a root but err > 1e-8, we might keep the best one as fallback?
        # But we are in the loop to find a GOOD root.
        # If we exhaust the search, we return the best candidate found so far.
        
        if root is not None and err < best_error:
            best_error = err
            best_candidate = root

    # Fallback: If polynomial reconstruction failed to find a root with |f(x)|<=1e-8
    # Use a robust hybrid method (Secant + Bisection) from x0
    if not check_deadline():
        return best_candidate
        
    # Simple Secant method fallback
    x_prev = x0
    fx_prev = safe_f(x_prev)
    if fx_prev is None:
        return best_candidate
        
    if abs(fx_prev) <= 1e-8:
        return x_prev
        
    delta = 1e-4 * max(1.0, abs(x_prev))
    x_curr = x_prev + delta
    fx_curr = safe_f(x_curr)
    
    if fx_curr is None:
        return best_candidate
        
    if abs(fx_curr) <= 1e-8:
        return x_curr
        
    best_x = x_curr if abs(fx_curr) < abs(fx_prev) else x_prev
    best_fx = fx_curr if abs(fx_curr) < abs(fx_prev) else fx_prev
    
    for _ in range(50):
        if not check_deadline():
            break
            
        if abs(fx_curr) <= 1e-8:
            return x_curr
        if abs(fx_prev) <= 1e-8:
            return x_prev
            
        if abs(fx_curr) < abs(best_fx):
            best_x = x_curr
            best_fx = fx_curr
        if abs(fx_prev) < abs(best_fx):
            best_x = x_prev
            best_fx = fx_prev
            
        if fx_prev == fx_curr:
            break
            
        denom = fx_curr - fx_prev
        if denom == 0 or not is_finite(denom):
            break
            
        x_next = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        if not is_finite(x_next):
            break
            
        # Dampen
        max_step = 10.0 * max(1.0, abs(x_curr))
        if abs(x_next - x_curr) > max_step:
            direction = 1 if x_next > x_curr else -1
            x_next = x_curr + direction * max_step
            
        fx_next = safe_f(x_next)
        
        if fx_next is None:
            break
            
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_next
        fx_curr = fx_next
        
    return best_x
# EVOLVE-BLOCK-END
