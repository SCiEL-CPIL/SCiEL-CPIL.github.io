"""Root finder for 3 fixed oscillatory equations with optimized call budget."""
import math
import time

# EVOLVE-BLOCK-START

def find_root(f, x0):
    """
    Find x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Local bracketing search with geometric steps.
    2. Hybrid Secant/Bisection convergence (Illinois-style logic).
    3. Steffensen's method with adaptive step sizing for high-precision polish.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    tol = 1e-8
    
    x0 = float(x0)
    
    # 1. Initial Evaluation
    try:
        f0 = f(x0)
    except Exception:
        # If f(x0) fails, try a slight offset
        x_try = x0 + 1e-6
        try:
            f0 = f(x_try)
            x0 = x_try
        except Exception:
            return x0  # Last resort
    
    if not math.isfinite(f0):
        return x0
        
    if abs(f0) <= tol:
        return x0

    # 2. Bracketing Search
    # We look for a sign change in [x0-d, x0] or [x0, x0+d]
    # We use a set of geometrically spaced distances
    deltas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    a = None
    b = None
    fa = None
    fb = None
    
    # We also keep track of the best point (smallest |f|) found
    best_x = x0
    best_f = f0
    best_abs = abs(f0)
    
    def update_best(x, fx):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(x) and math.isfinite(fx):
            af = abs(fx)
            if af < best_abs:
                best_x = x
                best_f = fx
                best_abs = af

    for d in deltas:
        if time.time() > deadline:
            break
            
        x_left = x0 - d
        x_right = x0 + d
        
        # Evaluate left
        try:
            fl = f(x_left)
        except Exception:
            fl = float('nan')
            
        if math.isfinite(fl):
            update_best(x_left, fl)
            if abs(fl) <= tol:
                return x_left
            if fa is None:
                fa = fl
                a = x_left
            # Check sign change with x0
            if fa * f0 < 0:
                # Found bracket [x_left, x0]
                a, fa = x_left, fl
                b, fb = x0, f0
                break
        
        # Evaluate right
        try:
            fr = f(x_right)
        except Exception:
            fr = float('nan')
            
        if math.isfinite(fr):
            update_best(x_right, fr)
            if abs(fr) <= tol:
                return x_right
            if fb is None:
                fb = fr
                b = x_right
            # Check sign change with x0
            if f0 * fr < 0:
                # Found bracket [x0, x_right]
                a, fa = x0, f0
                b, fb = x_right, fr
                break

    # If we found a bracket, use Hybrid Secant/Bisection
    if a is not None and b is not None:
        # Ensure a < b
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
            
        # Hybrid Iteration
        for _ in range(50):
            if time.time() > deadline:
                break
                
            # Check current endpoints
            if abs(fa) <= tol:
                return a
            if abs(fb) <= tol:
                return b
            
            # Calculate Secant Point
            denom = fb - fa
            if denom == 0 or not math.isfinite(denom):
                # Fallback to Bisection
                x_new = (a + b) / 2
            else:
                x_secant = b - fb * (b - a) / denom
                
                # Check if Secant point is inside bracket
                if a < x_secant < b:
                    x_new = x_secant
                else:
                    # If outside, use Bisection
                    x_new = (a + b) / 2
                    
            if not math.isfinite(x_new):
                break
                
            try:
                f_new = f(x_new)
            except Exception:
                break
                
            if not math.isfinite(f_new):
                break
                
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
                
            # Update Bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
            # Convergence check on x
            if abs(b - a) < 1e-15 * max(1.0, abs(b)):
                break
        
        # Polish pass: If we are close, try to get even closer using Steffensen's method
        if best_abs <= 1e-6:
            # Use best_x as center for Steffensen's method
            x = best_x
            fx = best_f
            
            for _ in range(20):
                if time.time() > deadline:
                    break
                
                if abs(fx) <= tol:
                    return x
                
                # Adaptive step size for Steffensen
                # Start with a small step, adjust if needed
                delta = max(1e-10, abs(fx) * 1e-6)
                
                x1 = x + delta
                try:
                    f1 = f(x1)
                except Exception:
                    break
                
                if not math.isfinite(f1):
                    break
                
                update_best(x1, f1)
                
                # Steffensen's iteration: x_{n+1} = x_n - f(x_n) / (f(x_n + delta) - f(x_n))
                # Wait, standard Steffensen is for fixed point g(x).
                # For root finding, it's essentially Newton's method with finite difference derivative.
                # f'(x) ~ (f(x+delta) - f(x)) / delta
                # x_new = x - f(x) / f'(x) = x - f(x) * delta / (f(x+delta) - f(x))
                
                denom = f1 - fx
                if abs(denom) < 1e-30:
                    # Derivative too small or zero, try bisection if we have a bracket
                    if a is not None and b is not None and min(a, b) <= x <= max(a, b):
                        # Fallback to bisection polish
                        pa, pb = a, b
                        pfa, pfb = fa, fb
                        for _ in range(25):
                            if time.time() > deadline:
                                break
                            if abs(pfa) <= tol:
                                return pa
                            if abs(pfb) <= tol:
                                return pb
                            
                            x_mid = (pa + pb) / 2
                            try:
                                f_mid = f(x_mid)
                            except Exception:
                                break
                            if not math.isfinite(f_mid):
                                break
                            update_best(x_mid, f_mid)
                            if abs(f_mid) <= tol:
                                return x_mid
                            
                            if pfa * f_mid < 0:
                                pb, pfb = x_mid, f_mid
                            else:
                                pa, pfa = x_mid, f_mid
                    break
                
                x_new = x - fx * delta / denom
                
                if not math.isfinite(x_new):
                    break
                
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                
                if not math.isfinite(f_new):
                    break
                
                update_best(x_new, f_new)
                
                if abs(f_new) <= tol:
                    return x_new
                
                x = x_new
                fx = f_new
        
        return best_x if best_abs <= tol else (a + b) / 2

    # No bracket found: fallback to secant
    x1 = x0
    f1 = f0
    
    # Find a second point
    candidates = [(x0 + 1e-2, None), (x0 - 1e-2, None), (x0 + 1.0, None), (x0 - 1.0, None)]
    
    x2 = None
    f2 = None
    min_abs = float('inf')
    
    for cx, _ in candidates:
        try:
            cf = f(cx)
        except Exception:
            continue
        if math.isfinite(cf):
            update_best(cx, cf)
            if abs(cf) < min_abs:
                min_abs = abs(cf)
                x2 = cx
                f2 = cf
    
    if x2 is None:
        # Fallback: try a larger step
        x2 = x0 + 10.0
        try:
            f2 = f(x2)
        except Exception:
            return best_x
        if not math.isfinite(f2):
            return best_x
        update_best(x2, f2)

    # Run Secant
    for _ in range(30):
        if time.time() > deadline:
            break
            
        if abs(f1) <= tol:
            return x1
        if abs(f2) <= tol:
            return x2
        
        denom = f2 - f1
        if denom == 0 or not math.isfinite(denom):
            break
            
        x_new = x2 - f2 * (x2 - x1) / denom
        
        if not math.isfinite(x_new):
            break
            
        try:
            f_new = f(x_new)
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        update_best(x_new, f_new)
        
        if abs(f_new) <= tol:
            return x_new
            
        x1, f1 = x2, f2
        x2, f2 = x_new, f_new
        
    # Polish pass for Secant case using Steffensen's method
    if best_abs <= 1e-6:
        x = best_x
        fx = best_f
        
        for _ in range(20):
            if time.time() > deadline:
                break
            
            if abs(fx) <= tol:
                return x
            
            # Adaptive step size
            delta = max(1e-10, abs(fx) * 1e-6)
            
            x1 = x + delta
            try:
                f1 = f(x1)
            except Exception:
                break
            
            if not math.isfinite(f1):
                break
            
            update_best(x1, f1)
            
            denom = f1 - fx
            if abs(denom) < 1e-30:
                break
            
            x_new = x - fx * delta / denom
            
            if not math.isfinite(x_new):
                break
            
            try:
                f_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(f_new):
                break
            
            update_best(x_new, f_new)
            
            if abs(f_new) <= tol:
                return x_new
            
            x = x_new
            fx = f_new

    return best_x

# EVOLVE-BLOCK-END
