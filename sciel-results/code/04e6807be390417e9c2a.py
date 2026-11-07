# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Strategy:
    1. Global Grid Search: Scan a logarithmic grid of points around x0 to find a sign change.
       This is robust to the location of the root relative to x0.
    2. Newton Polish: Once a bracket is found, use Newton's method (with bisection fallback
       if Newton steps are unstable or go out of bounds) to converge to the root quickly.
       Newton converges quadratically, reducing the number of calls compared to pure bisection/secant.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x0 = float(x0)
    if not math.isfinite(x0):
        return 0.0

    # Check if x0 is already a root
    f_x0 = f(x0)
    if abs(f_x0) <= 1e-8:
        return x0

    best_x = x0
    best_f = f_x0

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) <= 0
    # We search in a wide range using a logarithmic scale to handle roots far from x0.
    # We also check small intervals to handle roots very close to x0.
    
    a = None
    b = None
    fa = None
    fb = None
    
    # Define search steps: mix of small and large steps
    # Logarithmic steps: 10^-4 to 10^6
    steps = []
    for exp in range(-4, 7):
        steps.append(10**exp)
        steps.append(-10**exp)
    
    # Add some intermediate steps for better coverage
    steps.extend([0.1, -0.1, 0.5, -0.5, 2.0, -2.0, 5.0, -5.0, 20.0, -20.0, 50.0, -50.0, 200.0, -200.0, 500.0, -500.0])
    
    # Sort steps by absolute value to try small ones first? 
    # Actually, for polynomials, if x0 is close, small steps are best. If x0 is far, large steps are needed.
    # Let's try a mix. First try small, then large.
    # Re-sorting to prioritize smaller magnitudes first, as roots are often near x0 in these tasks.
    steps.sort(key=lambda s: abs(s))
    
    found_bracket = False
    
    for step in steps:
        if time.time() > deadline:
            return best_x
            
        b_candidate = x0 + step
        if not math.isfinite(b_candidate):
            continue
            
        fb_candidate = f(b_candidate)
        if not math.isfinite(fb_candidate):
            continue
            
        if abs(fb_candidate) <= 1e-8:
            return b_candidate
        
        # Update best
        if abs(fb_candidate) < abs(best_f):
            best_f = fb_candidate
            best_x = b_candidate

        # Check for sign change with x0
        if f_x0 * fb_candidate <= 0:
            a = x0
            fa = f_x0
            b = b_candidate
            fb = fb_candidate
            found_bracket = True
            break
        
        # Also check if we crossed a root between previous point and this one?
        # We don't have a previous point in this simple loop, so we just check against x0.
        # To be more robust, we could keep a sliding window, but for polynomials, 
        # checking against x0 is often sufficient if steps are dense enough near x0.
        # However, if x0 is not near the root, this might miss it if the function oscillates.
        # But the task says "smooth polynomial equations", implying single root or well-behaved.
        
    if not found_bracket:
        # Try a denser search in a larger range if needed
        # For now, if we haven't found it, we might have missed it.
        # Let's try a few more specific points if the first pass failed.
        # Actually, the sorted steps cover 1e-4 to 1e6. This should be sufficient for most cases.
        # If it fails, return best_x.
        return best_x

    # Phase 2: Polynomial Interpolation + Newton Polish
    # We have a bracket [a, b] with f(a)*f(b) <= 0.
    # Instead of iterative bisection/secant, we will:
    # 1. Sample 10 points in [a, b] (including endpoints).
    # 2. Fit a degree-9 polynomial using numpy.polyfit (or roots of Lagrange basis).
    # 3. Find roots of the polynomial using numpy.roots.
    # 4. Pick the real root closest to the midpoint (or within the bracket).
    # 5. Use that as a starting point for a few secant/Newton steps to ensure |f(x)| <= 1e-8.
    
    # Generate 10 points in [a, b]
    # Use linear spacing for stability in polynomial fitting
    a_min, a_max = min(a, b), max(a, b)
    # If the bracket is extremely small, just return the midpoint
    if a_max - a_min < 1e-10:
        mid = (a + b) / 2.0
        f_mid = f(mid)
        if abs(f_mid) <= 1e-8:
            return mid
        # If not, try one more secant step
        if abs(fa) < abs(fb):
            x_prev = a
            f_prev = fa
            x_curr = b
            f_curr = fb
        else:
            x_prev = b
            f_prev = fb
            x_curr = a
            f_curr = fa
            
        denom = f_curr - f_prev
        if abs(denom) < 1e-15:
            return mid
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        if math.isfinite(x_new):
            f_new = f(x_new)
            if math.isfinite(f_new) and abs(f_new) < abs(f_mid):
                return x_new
        return mid

    # Sample 10 points
    xs = np.linspace(a_min, a_max, 10)
    ys = np.array([f(x) for x in xs])
    
    # Check if any point is already a root
    for i in range(len(xs)):
        if abs(ys[i]) <= 1e-8:
            return xs[i]
        if abs(ys[i]) < abs(best_f):
            best_f = ys[i]
            best_x = xs[i]

    # Fit polynomial
    # We want to find roots of the interpolating polynomial.
    # np.polyfit returns coefficients for highest power first.
    # We can use np.roots to find the roots.
    
    try:
        # Use a high degree for better approximation, but 9 is fixed by 10 points
        coeffs = np.polyfit(xs, ys, 9)
        roots = np.roots(coeffs)
        
        # Filter for real roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-8]
        
        # Find the real root that is within the bracket [a_min, a_max]
        candidate = None
        min_dist = float('inf')
        mid = (a_min + a_max) / 2.0
        
        for r in real_roots:
            if a_min <= r <= a_max:
                # Prefer the one closest to the midpoint? Or just the first one?
                # Since the function is smooth and we have a sign change, there should be one root.
                # If there are multiple, the one closest to the midpoint is likely the one we want.
                dist = abs(r - mid)
                if dist < min_dist:
                    min_dist = dist
                    candidate = r
                    
        if candidate is not None:
            # We have a candidate root from the polynomial fit
            x_start = candidate
            f_start = f(x_start)
            
            if abs(f_start) <= 1e-8:
                return x_start
                
            if abs(f_start) < abs(best_f):
                best_f = f_start
                best_x = x_start
                
            # Now polish with Secant method
            # We need two points for Secant. We can use x_start and one of the sample points
            # Or just use x_start and x_start + epsilon?
            # Better: Use x_start and the point with the smallest |f| among the samples?
            # Or use the bracket endpoints a and b?
            # Let's use x_start and the sample point closest to x_start.
            
            # Find index of sample point closest to x_start
            idx = np.argmin(np.abs(xs - x_start))
            x_prev = xs[idx]
            f_prev = ys[idx]
            
            # If x_prev is same as x_start, pick another
            if abs(x_prev - x_start) < 1e-12:
                # Pick the other endpoint
                if idx == 0:
                    x_prev = xs[1]
                    f_prev = ys[1]
                else:
                    x_prev = xs[idx-1]
                    f_prev = ys[idx-1]
                    
            x_curr = x_start
            f_curr = f_start
            
            max_iter = 20
            for i in range(max_iter):
                if time.time() > deadline:
                    return best_x
                    
                if abs(f_curr) <= 1e-8:
                    return x_curr
                if abs(f_prev) <= 1e-8:
                    return x_prev
                    
                if abs(f_curr) < abs(best_f):
                    best_f = f_curr
                    best_x = x_curr
                if abs(f_prev) < abs(best_f):
                    best_f = f_prev
                    best_x = x_prev
                    
                denom = f_curr - f_prev
                if abs(denom) < 1e-15:
                    # Degenerate, break
                    break
                
                x_new = x_curr - f_curr * (x_curr - x_prev) / denom
                
                if not math.isfinite(x_new):
                    break
                    
                f_new = f(x_new)
                if not math.isfinite(f_new):
                    break
                    
                if abs(f_new) <= 1e-8:
                    return x_new
                    
                if abs(f_new) < abs(best_f):
                    best_f = f_new
                    best_x = x_new
                    
                # Update Secant points
                # Keep the two most recent points? Or the two with smallest |f|?
                # Standard Secant uses the two most recent.
                x_prev = x_curr
                f_prev = f_curr
                x_curr = x_new
                f_curr = f_new
                
            return best_x
            
    except Exception:
        pass

    # Fallback to Bisection if interpolation failed or didn't yield a good candidate
    # We have [a, b] with fa, fb
    x0_new = a
    x1_new = b
    f0 = fa
    f1 = fb
    
    max_iter = 100
    for i in range(max_iter):
        if time.time() > deadline:
            return best_x
            
        # Check if we have a root
        if abs(f0) <= 1e-8:
            return x0_new
        if abs(f1) <= 1e-8:
            return x1_new
            
        # Update best
        if abs(f0) < abs(best_f):
            best_f = f0
            best_x = x0_new
        if abs(f1) < abs(best_f):
            best_f = f1
            best_x = x1_new

        # Bisection step
        x_new = (x0_new + x1_new) / 2.0
        f_new = f(x_new)
        if not math.isfinite(f_new):
            break
            
        if abs(f_new) <= 1e-8:
            return x_new
            
        # Update best
        if abs(f_new) < abs(best_f):
            best_f = f_new
            best_x = x_new

        # Update bracket
        if f0 * f_new <= 0:
            x1_new = x_new
            f1 = f_new
        else:
            x0_new = x_new
            f0 = f_new
            
    return best_x

# EVOLVE-BLOCK-END
