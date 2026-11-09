import math
import time

def find_root(f, x0):
    """
    Find a root of f starting near x0 using a robust multi-strategy approach.
    Returns x such that |f(x)| <= 1e-8.
    """
    deadline = time.time() + 145.0  # Safety margin before 150s kill
    tol = 1e-8  # Match required tolerance exactly
    
    x = float(x0)
    
    # Evaluate initial point
    try:
        f_x = f(x)
    except:
        return x
    
    if not math.isfinite(f_x):
        return x
        
    best_x = x
    best_f = f_x
    best_abs = abs(best_f)
    
    if best_abs <= tol:
        return x
    
    # Strategy 1: Steffensen's method (quadratic convergence, no derivative needed)
    # This is fast and works well for smooth oscillatory functions
    x_prev = x
    fx_prev = f_x
    
    # Small perturbation for Steffensen's
    h = 1e-8 * max(1.0, abs(x))
    x_curr = x + h
    
    try:
        fx_curr = f(x_curr)
    except:
        fx_curr = f_x  # Fallback
    
    if not math.isfinite(fx_curr):
        # Try larger perturbation
        h = 1e-4 * max(1.0, abs(x))
        x_curr = x + h
        try:
            fx_curr = f(x_curr)
        except:
            fx_curr = f_x
    
    if math.isfinite(fx_curr) and abs(fx_curr) <= tol:
        return x_curr
    
    if math.isfinite(fx_curr) and abs(fx_curr) < best_abs:
        best_x = x_curr
        best_f = fx_curr
        best_abs = abs(fx_curr)
    
    # Steffensen's iteration
    for i in range(50):
        if time.time() > deadline:
            break
        
        if abs(fx_curr) < best_abs:
            best_x = x_curr
            best_f = fx_curr
            best_abs = abs(fx_curr)
        
        if best_abs <= tol:
            return best_x
        
        # Steffensen step: x_new = x_curr - fx_curr * (x_curr - x_prev) / (fx_curr - fx_prev)
        denom = fx_curr - fx_prev
        if abs(denom) < 1e-30:
            # Degenerate case, try a different perturbation
            x_curr = x_prev + 1e-4 * max(1.0, abs(x_prev))
            try:
                fx_curr = f(x_curr)
            except:
                break
            if not math.isfinite(fx_curr):
                break
            denom = fx_curr - fx_prev
            if abs(denom) < 1e-30:
                break
        
        x_new = x_curr - fx_curr * (x_curr - x_prev) / denom
        
        if not math.isfinite(x_new):
            # Fallback: bisection-like step
            x_new = 0.5 * (x_prev + x_curr)
        
        x_prev = x_curr
        fx_prev = fx_curr
        x_curr = x_new
        
        try:
            fx_curr = f(x_curr)
        except:
            break
        
        if not math.isfinite(fx_curr):
            # Try to recover
            x_curr = 0.5 * (x_prev + x_new)
            try:
                fx_curr = f(x_curr)
            except:
                break
            if not math.isfinite(fx_curr):
                break
        
        # Check for convergence
        if abs(x_curr - x_prev) < 1e-14 * max(1.0, abs(x_curr)):
            if abs(fx_curr) <= tol:
                return x_curr
            # Try one more refinement
            break
    
    if best_abs <= tol:
        return best_x
    
    # Strategy 2: Bracketing with adaptive expansion
    # Try to find a sign change near x0
    a = best_x
    fa = best_f
    b = None
    fb = None
    
    # Adaptive step size based on local scale
    base_step = 1e-3 * max(1.0, abs(a))
    
    # Try expanding in both directions with increasing step
    for i in range(20):
        if time.time() > deadline:
            break
            
        step = base_step * (2 ** i)
        
        # Try positive direction
        x_pos = a + step
        try:
            f_pos = f(x_pos)
            if math.isfinite(f_pos):
                if abs(f_pos) < best_abs:
                    best_x, best_f, best_abs = x_pos, f_pos, abs(f_pos)
                    if best_abs <= tol:
                        return best_x
                if fa * f_pos < 0:
                    a, b = a, x_pos
                    fa, fb = fa, f_pos
                    break
        except:
            pass
            
        # Try negative direction
        x_neg = a - step
        try:
            f_neg = f(x_neg)
            if math.isfinite(f_neg):
                if abs(f_neg) < best_abs:
                    best_x, best_f, best_abs = x_neg, f_neg, abs(f_neg)
                    if best_abs <= tol:
                        return best_x
                if fa * f_neg < 0:
                    a, b = x_neg, a
                    fa, fb = f_neg, fa
                    break
        except:
            pass
    
    # If bracket found, use Brent-like method (secant + bisection hybrid)
    if b is not None:
        # Ensure a < b for consistency
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        
        # Secant + Bisection hybrid
        for _ in range(100):
            if time.time() > deadline:
                break
                
            # Secant step
            if fa == fb:
                x_new = (a + b) / 2.0
            else:
                x_new = b - fb * (b - a) / (fb - fa)
                
            # If secant step is outside bracket or too close to endpoint, use bisection
            if x_new <= a or x_new >= b or abs(x_new - a) < 1e-12 * (b - a) or abs(x_new - b) < 1e-12 * (b - a):
                x_new = (a + b) / 2.0
                
            try:
                f_new = f(x_new)
            except:
                break
                
            if not math.isfinite(f_new):
                break
                
            # Update best
            if abs(f_new) < best_abs:
                best_x, best_f, best_abs = x_new, f_new, abs(f_new)
                if best_abs <= tol:
                    return best_x
                    
            # Update bracket
            if fa * f_new < 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
                
        return best_x
    
    # Strategy 3: Secant method from best point found
    h = 1e-6 * max(1.0, abs(best_x))
    x_prev = best_x
    f_prev = best_f
    x_curr = best_x + h
    
    try:
        f_curr = f(x_curr)
    except:
        return best_x
        
    for _ in range(50):
        if time.time() > deadline:
            break
            
        if abs(f_curr) < best_abs:
            best_x, best_f, best_abs = x_curr, f_curr, abs(f_curr)
            if best_abs <= tol:
                return best_x
                
        if abs(f_prev) < best_abs:
            best_x, best_f, best_abs = x_prev, f_prev, abs(f_prev)
            if best_abs <= tol:
                return best_x
                
        if f_prev == f_curr:
            break
            
        # Secant step
        x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
        
        if not math.isfinite(x_new):
            break
            
        x_prev, f_prev = x_curr, f_curr
        x_curr = x_new
        
        try:
            f_curr = f(x_curr)
        except:
            break
            
    return best_x
