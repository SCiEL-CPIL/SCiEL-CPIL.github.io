# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0 using a Brent's method-like approach with safeguards.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s)
    tol = 1e-8
    
    x = float(x0)
    
    # Initial bracketing attempt
    h = 1e-4 * max(1.0, abs(x))
    
    try:
        f_x = f(x)
    except Exception:
        return x
    
    if not math.isfinite(f_x):
        return x
    
    if abs(f_x) <= tol:
        return x
    
    # Try to find a bracket
    a, b = x, x + h
    fa = f_x
    try:
        fb = f(b)
    except Exception:
        fb = None
    
    if fb is not None and math.isfinite(fb):
        if abs(fb) <= tol:
            return b
        # Check if we have a sign change
        if fa * fb < 0:
            # We have a bracket, use bisection/secant hybrid
            best_x = a if abs(fa) < abs(fb) else b
            best_f = min(abs(fa), abs(fb))
            
            for _ in range(100):
                if time.time() > deadline:
                    break
                
                if abs(fb) <= tol:
                    return b
                if abs(fa) <= tol:
                    return a
                
                # Secant step
                if fa != fb:
                    x_new = b - fb * (b - a) / (fb - fa)
                else:
                    x_new = (a + b) / 2
                
                # Check if secant step is within bracket
                if not (a < x_new < b) or not math.isfinite(x_new):
                    x_new = (a + b) / 2  # Bisection fallback
                
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                
                if not math.isfinite(f_new):
                    break
                
                if abs(f_new) <= tol:
                    return x_new
                
                if abs(f_new) < best_f:
                    best_x = x_new
                    best_f = abs(f_new)
                
                # Update bracket
                if fa * f_new < 0:
                    b = x_new
                    fb = f_new
                else:
                    a = x_new
                    fa = f_new
                
                if abs(b - a) < 1e-12 * max(1.0, abs(b)):
                    break
            
            # Final polish with Newton's method if we're close
            if best_f > tol and best_f < 1e-4:
                for _ in range(10):
                    if time.time() > deadline:
                        break
                    h_polish = 1e-6 * max(1.0, abs(best_x))
                    try:
                        f_l = f(best_x - h_polish)
                        f_r = f(best_x + h_polish)
                        if math.isfinite(f_l) and math.isfinite(f_r):
                            deriv = (f_r - f_l) / (2 * h_polish)
                            if deriv != 0 and math.isfinite(deriv):
                                f_best = f(best_x)
                                if math.isfinite(f_best):
                                    x_polish = best_x - f_best / deriv
                                    if math.isfinite(x_polish):
                                        f_polish = f(x_polish)
                                        if math.isfinite(f_polish) and abs(f_polish) < best_f:
                                            best_x = x_polish
                                            best_f = abs(f_polish)
                                        if abs(f_polish) <= tol:
                                            return x_polish
                    except Exception:
                        break
            
            return best_x
        
        # No sign change, try expanding the bracket
        for _ in range(20):
            h *= 2
            b = x + h
            try:
                fb = f(b)
            except Exception:
                break
            
            if not math.isfinite(fb):
                break
            
            if abs(fb) <= tol:
                return b
            
            if fa * fb < 0:
                # Found a bracket, use bisection/secant hybrid
                a = x
                best_x = a if abs(fa) < abs(fb) else b
                best_f = min(abs(fa), abs(fb))
                
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    if abs(fb) <= tol:
                        return b
                    if abs(fa) <= tol:
                        return a
                    
                    # Secant step
                    if fa != fb:
                        x_new = b - fb * (b - a) / (fb - fa)
                    else:
                        x_new = (a + b) / 2
                    
                    # Check if secant step is within bracket
                    if not (a < x_new < b) or not math.isfinite(x_new):
                        x_new = (a + b) / 2  # Bisection fallback
                    
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    
                    if not math.isfinite(f_new):
                        break
                    
                    if abs(f_new) <= tol:
                        return x_new
                    
                    if abs(f_new) < best_f:
                        best_x = x_new
                        best_f = abs(f_new)
                    
                    # Update bracket
                    if fa * f_new < 0:
                        b = x_new
                        fb = f_new
                    else:
                        a = x_new
                        fa = f_new
                    
                    if abs(b - a) < 1e-12 * max(1.0, abs(b)):
                        break
                
                # Final polish with Newton's method if we're close
                if best_f > tol and best_f < 1e-4:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        h_polish = 1e-6 * max(1.0, abs(best_x))
                        try:
                            f_l = f(best_x - h_polish)
                            f_r = f(best_x + h_polish)
                            if math.isfinite(f_l) and math.isfinite(f_r):
                                deriv = (f_r - f_l) / (2 * h_polish)
                                if deriv != 0 and math.isfinite(deriv):
                                    f_best = f(best_x)
                                    if math.isfinite(f_best):
                                        x_polish = best_x - f_best / deriv
                                        if math.isfinite(x_polish):
                                            f_polish = f(x_polish)
                                            if math.isfinite(f_polish) and abs(f_polish) < best_f:
                                                best_x = x_polish
                                                best_f = abs(f_polish)
                                            if abs(f_polish) <= tol:
                                                return x_polish
                        except Exception:
                            break
                
                return best_x
        
        # If we couldn't bracket, fall back to secant method without bracket
        best_x = x
        best_f = abs(f_x)
        
        h = 1e-4 * max(1.0, abs(x))
        x_next = x + h
        try:
            f_next = f(x_next)
        except Exception:
            return best_x
        
        if not math.isfinite(f_next):
            return best_x
        
        if abs(f_next) <= tol:
            return x_next
        
        if abs(f_next) < best_f:
            best_x = x_next
            best_f = abs(f_next)
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            if f_x == f_next:
                # Degenerate case, perturb x_next
                x_next = x_next + 1e-4 * max(1.0, abs(x_next))
                try:
                    f_next = f(x_next)
                except Exception:
                    break
                if not math.isfinite(f_next):
                    break
                if abs(f_next) <= tol:
                    return x_next
                if abs(f_next) < best_f:
                    best_x = x_next
                    best_f = abs(f_next)
                continue
            
            x_new = x_next - f_next * (x_next - x) / (f_next - f_x)
            
            if not math.isfinite(x_new):
                break
            
            # Safeguard against divergence
            if abs(x_new - x_next) > 10 * max(1.0, abs(x_next)):
                # Try a damped Newton step using the secant slope as derivative estimate
                slope = (f_next - f_x) / (x_next - x)
                if slope != 0 and math.isfinite(slope):
                    x_new_damped = x_next - f_next / slope
                    if math.isfinite(x_new_damped):
                        x_new = x_new_damped
                    else:
                        break
                else:
                    break
            
            try:
                f_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(f_new):
                break
            
            if abs(f_new) <= tol:
                return x_new
            
            if abs(f_new) < best_f:
                best_x = x_new
                best_f = abs(f_new)
                
            x, f_x = x_next, f_next
            x_next, f_next = x_new, f_new
        
        # Final polish pass
        if best_f > tol and best_f < 1e-4:
            for _ in range(10):
                if time.time() > deadline:
                    break
                h_polish = 1e-6 * max(1.0, abs(best_x))
                try:
                    f_l = f(best_x - h_polish)
                    f_r = f(best_x + h_polish)
                    if math.isfinite(f_l) and math.isfinite(f_r):
                        deriv = (f_r - f_l) / (2 * h_polish)
                        if deriv != 0 and math.isfinite(deriv):
                            f_best = f(best_x)
                            if math.isfinite(f_best):
                                x_polish = best_x - f_best / deriv
                                if math.isfinite(x_polish):
                                    f_polish = f(x_polish)
                                    if math.isfinite(f_polish) and abs(f_polish) < best_f:
                                        best_x = x_polish
                                        best_f = abs(f_polish)
                                    if abs(f_polish) <= tol:
                                        return x_polish
                except Exception:
                    break
        
        return best_x
    else:
        # fb is not finite, try smaller h
        for h_factor in [1e-6, 1e-8]:
            h = h_factor * max(1.0, abs(x))
            x_next = x + h
            try:
                f_next = f(x_next)
            except Exception:
                continue
            
            if not math.isfinite(f_next):
                continue
            
            if abs(f_next) <= tol:
                return x_next
            
            if f_x * f_next < 0:
                # Found a bracket
                a, b = x, x_next
                fa, fb = f_x, f_next
                best_x = a if abs(fa) < abs(fb) else b
                best_f = min(abs(fa), abs(fb))
                
                for _ in range(100):
                    if time.time() > deadline:
                        break
                    
                    if abs(fb) <= tol:
                        return b
                    if abs(fa) <= tol:
                        return a
                    
                    # Secant step
                    if fa != fb:
                        x_new = b - fb * (b - a) / (fb - fa)
                    else:
                        x_new = (a + b) / 2
                    
                    # Check if secant step is within bracket
                    if not (a < x_new < b) or not math.isfinite(x_new):
                        x_new = (a + b) / 2  # Bisection fallback
                    
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    
                    if not math.isfinite(f_new):
                        break
                    
                    if abs(f_new) <= tol:
                        return x_new
                    
                    if abs(f_new) < best_f:
                        best_x = x_new
                        best_f = abs(f_new)
                    
                    # Update bracket
                    if fa * f_new < 0:
                        b = x_new
                        fb = f_new
                    else:
                        a = x_new
                        fa = f_new
                    
                    if abs(b - a) < 1e-12 * max(1.0, abs(b)):
                        break
                
                # Final polish with Newton's method if we're close
                if best_f > tol and best_f < 1e-4:
                    for _ in range(10):
                        if time.time() > deadline:
                            break
                        h_polish = 1e-6 * max(1.0, abs(best_x))
                        try:
                            f_l = f(best_x - h_polish)
                            f_r = f(best_x + h_polish)
                            if math.isfinite(f_l) and math.isfinite(f_r):
                                deriv = (f_r - f_l) / (2 * h_polish)
                                if deriv != 0 and math.isfinite(deriv):
                                    f_best = f(best_x)
                                    if math.isfinite(f_best):
                                        x_polish = best_x - f_best / deriv
                                        if math.isfinite(x_polish):
                                            f_polish = f(x_polish)
                                            if math.isfinite(f_polish) and abs(f_polish) < best_f:
                                                best_x = x_polish
                                                best_f = abs(f_polish)
                                            if abs(f_polish) <= tol:
                                                return x_polish
                        except Exception:
                            break
                
                return best_x
        
        # Fall back to secant without bracket
        h = 1e-4 * max(1.0, abs(x))
        x_next = x + h
        try:
            f_next = f(x_next)
        except Exception:
            return x
        
        if not math.isfinite(f_next):
            return x
        
        if abs(f_next) <= tol:
            return x_next
        
        best_x = x if abs(f_x) < abs(f_next) else x_next
        best_f = min(abs(f_x), abs(f_next))
        
        for _ in range(100):
            if time.time() > deadline:
                break
            
            if f_x == f_next:
                # Degenerate case, perturb x_next
                x_next = x_next + 1e-4 * max(1.0, abs(x_next))
                try:
                    f_next = f(x_next)
                except Exception:
                    break
                if not math.isfinite(f_next):
                    break
                if abs(f_next) <= tol:
                    return x_next
                if abs(f_next) < best_f:
                    best_x = x_next
                    best_f = abs(f_next)
                continue
            
            x_new = x_next - f_next * (x_next - x) / (f_next - f_x)
            
            if not math.isfinite(x_new):
                break
            
            # Safeguard against divergence
            if abs(x_new - x_next) > 10 * max(1.0, abs(x_next)):
                # Try a damped Newton step using the secant slope as derivative estimate
                slope = (f_next - f_x) / (x_next - x)
                if slope != 0 and math.isfinite(slope):
                    x_new_damped = x_next - f_next / slope
                    if math.isfinite(x_new_damped):
                        x_new = x_new_damped
                    else:
                        break
                else:
                    break
            
            try:
                f_new = f(x_new)
            except Exception:
                break
            
            if not math.isfinite(f_new):
                break
            
            if abs(f_new) <= tol:
                return x_new
            
            if abs(f_new) < best_f:
                best_x = x_new
                best_f = abs(f_new)
                
            x, f_x = x_next, f_next
            x_next, f_next = x_new, f_new
        
        # Final polish pass
        if best_f > tol and best_f < 1e-4:
            for _ in range(10):
                if time.time() > deadline:
                    break
                h_polish = 1e-6 * max(1.0, abs(best_x))
                try:
                    f_l = f(best_x - h_polish)
                    f_r = f(best_x + h_polish)
                    if math.isfinite(f_l) and math.isfinite(f_r):
                        deriv = (f_r - f_l) / (2 * h_polish)
                        if deriv != 0 and math.isfinite(deriv):
                            f_best = f(best_x)
                            if math.isfinite(f_best):
                                x_polish = best_x - f_best / deriv
                                if math.isfinite(x_polish):
                                    f_polish = f(x_polish)
                                    if math.isfinite(f_polish) and abs(f_polish) < best_f:
                                        best_x = x_polish
                                        best_f = abs(f_polish)
                                    if abs(f_polish) <= tol:
                                        return x_polish
                except Exception:
                    break
        
        return best_x
# EVOLVE-BLOCK-END
