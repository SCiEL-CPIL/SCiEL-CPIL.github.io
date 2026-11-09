import numpy as np
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin before 120s limit
    
    x0 = float(x0)
    
    # If x0 is already a root
    fx0 = f(x0)
    if abs(fx0) <= 1e-8:
        return x0

    # Strategy: Polynomial Interpolation
    # Try increasing degrees to find the correct polynomial
    
    # Scale for step size to handle large/small x0
    scale = max(1.0, abs(x0))
    
    # We will collect points and values
    # Start with a base set of points centered around x0
    # We want to capture the local polynomial behavior.
    
    # Let's try degrees 3, 4, 5, 6, 7
    # For degree D, we need D+1 points.
    # We can reuse points from previous attempts.
    
    # Define a set of offsets. We'll use symmetric offsets for stability.
    # Offsets: -3, -2, -1, 0, 1, 2, 3, 4
    # This allows fitting up to degree 7 with 8 points.
    offsets = [-3, -2, -1, 0, 1, 2, 3, 4]
    
    xs = []
    ys = []
    
    # Evaluate points incrementally
    # We need to be careful with the call budget.
    # 8 calls for points + 1 for x0 (already done) + polish calls.
    
    # Let's evaluate the points
    for off in offsets:
        if time.time() > deadline:
            break
        xi = x0 + off * scale
        yi = f(xi)
        xs.append(xi)
        ys.append(yi)
        
    # If we ran out of time, return best guess (x0)
    if not xs:
        return x0

    # Fit polynomial of varying degrees
    best_x = x0
    best_fval = abs(fx0)
    
    # Try degrees from 3 up to len(xs)-1
    # Note: numpy.polyfit returns highest degree coefficient first
    for deg in range(3, len(xs)):
        if time.time() > deadline:
            break
            
        try:
            # Use polyfit for numerical stability
            coeffs = np.polyfit(xs, ys, deg)
            
            # Check residual to see if degree is sufficient
            # Evaluate the fitted polynomial at the xs
            p = np.poly1d(coeffs)
            residuals = p(xs) - np.array(ys)
            max_res = np.max(np.abs(residuals))
            
            # If residual is small relative to y scale, the fit is good
            # Scale of y: max(abs(ys))
            y_scale = max(1.0, np.max(np.abs(ys)))
            
            if max_res < 1e-6 * y_scale:
                # Find roots
                roots = np.roots(coeffs)
                
                # Filter real roots
                real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
                
                if real_roots:
                    # Find root closest to x0
                    closest_root = min(real_roots, key=lambda r: abs(r - x0))
                    
                    # Verify
                    f_val = f(closest_root)
                    
                    if abs(f_val) <= 1e-8:
                        return closest_root
                    else:
                        # Polish with Newton's method
                        # We have a good starting point
                        x = closest_root
                        for _ in range(5):
                            if time.time() > deadline:
                                break
                            fx = f(x)
                            if abs(fx) <= 1e-12:
                                return x
                            # Derivative from polynomial coeffs
                            # p'(x)
                            dp = np.polyder(p)
                            dx = dp(x)
                            if dx == 0:
                                break
                            x_new = x - fx / dx
                            if x_new == x:
                                break
                            x = x_new
                        if abs(f(x)) <= 1e-8:
                            return x
                        else:
                            # Update best if this is better
                            if abs(f(x)) < best_fval:
                                best_x = x
                                best_fval = abs(f(x))
                                
        except Exception:
            continue

    # If interpolation didn't yield a good root, fall back to robust iterative method
    # Use Secant or Bisection
    # We have points in xs, ys. Find a sign change.
    
    # Sort points by x
    sorted_points = sorted(zip(xs, ys))
    x_sorted = [p[0] for p in sorted_points]
    y_sorted = [p[1] for p in sorted_points]
    
    # Find interval with sign change
    a = None
    b = None
    for i in range(len(x_sorted) - 1):
        if y_sorted[i] * y_sorted[i+1] < 0:
            a = x_sorted[i]
            b = x_sorted[i+1]
            break
        elif y_sorted[i] == 0:
            return x_sorted[i]
            
    if a is None:
        # No sign change found in our sample points.
        # Try expanding search range or use x0 as best guess
        # If we have a candidate from interpolation, use it
        if best_fval < 1e-4:
             # Try to polish best_x further
             x = best_x
             for _ in range(10):
                 if time.time() > deadline:
                     break
                 fx = f(x)
                 if abs(fx) <= 1e-12:
                     return x
                 h = 1e-6 * max(1.0, abs(x))
                 d = (f(x + h) - fx) / h
                 if d == 0:
                     break
                 x = x - fx / d
             return x
        return best_x

    # Bisection on [a, b]
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
         # Should not happen if we found sign change, but check for 0
         if abs(fa) <= 1e-8: return a
         if abs(fb) <= 1e-8: return b
         
    for _ in range(60):
        if time.time() > deadline:
            break
        m = (a + b) / 2.0
        fm = f(m)
        if abs(fm) <= 1e-12:
            return m
        if fa * fm <= 0:
            b = m
            fb = fm
        else:
            a = m
            fa = fm
        if abs(b - a) < 1e-15 * max(1.0, abs(m)):
            break
            
    return (a + b) / 2.0
