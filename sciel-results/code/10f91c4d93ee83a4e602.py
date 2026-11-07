"""Root finder using inverse interpolation with Newton/Regula Falsi safeguard."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Deadline guard: return best result well before 120s limit
    deadline = time.time() + 110.0
    TOL = 1e-10  # tighter than 1e-8 for safety
    
    x = float(x0)
    fx = f(x)
    calls = 1
    
    if abs(fx) <= TOL:
        return x
    
    # Phase 1: Build a bracket around x0 with minimal extra calls
    # Try a small step to get a second point
    step = 1e-4 * max(1.0, abs(x))
    x2 = x + step
    fx2 = f(x2)
    calls += 1
    
    if abs(fx2) <= TOL:
        return x2
    
    # If we have a sign change, use bracketed method
    # If not, continue with unbracketed interpolation
    if fx * fx2 < 0:
        a, fa = x, fx
        b, fb = x2, fx2
        # Ensure |fa| <= |fb| so we keep the better point
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        best_x, best_fx = (a if abs(fa) < abs(fb) else b), (fa if abs(fa) < abs(fb) else fb)
        
        for i in range(40):
            if time.time() > deadline:
                return best_x
            if abs(best_fx) <= TOL:
                return best_x
            
            # Inverse quadratic interpolation if we have 3 points
            # For now use Regula Falsi (linear inverse interpolation)
            # which reuses the two bracket endpoints
            if fa * fb >= 0:
                # Shouldn't happen if bracket maintained, but safeguard
                break
            
            # Regula Falsi step
            if abs(fb) < abs(fa):
                a, b = b, a
                fa, fb = fb, fa
            
            # x_new = b - fb*(b-a)/(fb-fa)
            denom = fb - fa
            if abs(denom) < 1e-300:
                break
            x_new = b - fb * (b - a) / denom
            
            # Safeguard: if x_new is too close to a, bisect
            if not (a < x_new < b):
                x_new = 0.5 * (a + b)
            else:
                # Regula falsi can stall; enforce bisection if progress is slow
                if abs(x_new - b) < 0.1 * abs(b - a):
                    x_new = 0.5 * (a + b)
            
            fx_new = f(x_new)
            calls += 1
            
            if abs(fx_new) < abs(best_fx):
                best_x, best_fx = x_new, fx_new
            
            if abs(fx_new) <= TOL:
                return x_new
            
            # Update bracket
            if fa * fx_new <= 0:
                b, fb = x_new, fx_new
            else:
                a, fa = x_new, fx_new
            
            if abs(b - a) < 1e-15 * max(1.0, abs(x_new)):
                break
        
        return best_x
    
    # Phase 2: No bracket found. Use secant/Newton-like iteration.
    # We have (x, fx) and (x2, fx2). Use secant method with safeguards.
    a, fa = x, fx
    b, fb = x2, fx2
    best_x, best_fx = (a if abs(fa) < abs(fb) else b), (fa if abs(fa) < abs(fb) else fb)
    
    for i in range(40):
        if time.time() > deadline:
            return best_x
        if abs(best_fx) <= TOL:
            return best_x
        
        denom = fb - fa
        if abs(denom) < 1e-300:
            # Degenerate; try a small perturbation
            x_new = a + 1e-6 * max(1.0, abs(a))
        else:
            x_new = b - fb * (b - a) / denom
        
        # Safeguard against divergence: keep x_new within reasonable range
        if not math.isfinite(x_new):
            x_new = 0.5 * (a + b)
        
        # Limit step size to prevent runaway
        max_step = 1e6 * max(1.0, abs(a), abs(b))
        if abs(x_new - b) > max_step:
            x_new = b + math.copysign(max_step, x_new - b)
        
        fx_new = f(x_new)
        calls += 1
        
        if not math.isfinite(fx_new):
            # Move back toward the better point
            x_new = 0.5 * (a + b)
            fx_new = f(x_new)
            calls += 1
            if not math.isfinite(fx_new):
                return best_x
        
        if abs(fx_new) < abs(best_fx):
            best_x, best_fx = x_new, fx_new
        
        if abs(fx_new) <= TOL:
            return x_new
        
        # Update: keep the two points with smallest |f|
        if abs(fx_new) < abs(fa):
            a, fa = b, fb
        else:
            b, fb = x_new, fx_new
    
    return best_x
# EVOLVE-BLOCK-END
