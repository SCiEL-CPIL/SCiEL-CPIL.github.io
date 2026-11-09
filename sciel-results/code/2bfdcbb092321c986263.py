import math
import time
import numpy as np

def find_root(f, x0):
    tol = 1e-8
    deadline = time.time() + 115.0  # Safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= tol:
        return x
    
    # Try to find a bracket [a, b] where f(a) * f(b) < 0
    a = x
    fa = fx
    
    bracket_found = False
    b = None
    fb = None
    
    # First check nearby points for a sign change with geometric expansion
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    for delta in deltas:
        if time.time() > deadline:
            return x
        
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            b = x_plus
            fb = f_plus
            bracket_found = True
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            b = x_minus
            fb = f_minus
            bracket_found = True
            break
    
    # If no bracket found with initial deltas, try larger geometric expansion
    if not bracket_found:
        step = 1.0
        for i in range(40):
            if time.time() > deadline:
                return x
            
            # Try expanding in both directions
            for sign in [1, -1]:
                x_try = x + sign * step
                f_try = f(x_try)
                if abs(f_try) <= tol:
                    return x_try
                if fa * f_try < 0:
                    b = x_try
                    fb = f_try
                    bracket_found = True
                    break
            if bracket_found:
                break
            step *= 2.0
            # Cap the step to avoid extremely large values
            if step > 1e6:
                break
    
    if not bracket_found:
        # No bracket found; try a moderate grid scan (reduced cost)
        if time.time() < deadline - 5.0:
            # Try scanning a wider range with a grid
            grid_start = x - 100.0
            grid_end = x + 100.0
            n_points = 2000
            grid = np.linspace(grid_start, grid_end, n_points)
            
            # Evaluate f on the grid
            try:
                f_vals = np.array([f(xi) for xi in grid])
            except:
                f_vals = None
            
            if f_vals is not None and np.all(np.isfinite(f_vals)):
                # Find sign changes
                signs = np.sign(f_vals)
                sign_changes = np.where(signs[:-1] * signs[1:] < 0)[0]
                
                if len(sign_changes) > 0:
                    # Take the first sign change
                    idx = sign_changes[0]
                    a = grid[idx]
                    b = grid[idx + 1]
                    fa = f_vals[idx]
                    fb = f_vals[idx + 1]
                    bracket_found = True
        
        if not bracket_found:
            # Return the point with smallest |f|
            return x
    
    # Hybrid Secant/Bisection on [a, b]
    lo = a
    hi = b
    flo = fa
    fhi = fb
    
    # Ensure a < b for clarity
    if lo > hi:
        lo, hi = hi, lo
        flo, fhi = fhi, flo
    
    # Track best point found so far
    best_x = (lo + hi) / 2.0
    best_f = min(abs(flo), abs(fhi))
    if abs(flo) <= abs(fhi):
        best_x = lo
        best_f = abs(flo)
    else:
        best_x = hi
        best_f = abs(fhi)
    
    # Hybrid Solver: Aggressive Secant with Bisection Fallback
    for i in range(80):
        if time.time() > deadline:
            return best_x if best_f <= tol else (lo + hi) / 2.0
            
        if abs(hi - lo) < 1e-14:
            break
            
        # Attempt Secant Step (more aggressive)
        denom = fhi - flo
        if abs(denom) > 1e-300:
            x_secant = hi - fhi * (hi - lo) / denom
            
            # Allow secant step if it's not too far outside the bracket
            # and if it's likely to reduce |f|
            if lo - 0.5*(hi-lo) < x_secant < hi + 0.5*(hi-lo):
                x_new = x_secant
            else:
                x_new = (lo + hi) / 2.0
        else:
            x_new = (lo + hi) / 2.0
            
        f_new = f(x_new)
        
        if abs(f_new) <= tol:
            return x_new
        
        # Update best
        if abs(f_new) < best_f:
            best_f = abs(f_new)
            best_x = x_new
            
        # Update Bracket
        if flo * f_new < 0:
            hi = x_new
            fhi = f_new
        else:
            lo = x_new
            flo = f_new
            
    # Final polish: if we have a good bracket, do a few more bisection steps
    # to ensure |f(x)| <= tol
    if best_f > tol:
        # Do additional bisection steps to tighten the bracket
        for i in range(30):
            if time.time() > deadline:
                break
            mid = (lo + hi) / 2.0
            f_mid = f(mid)
            if abs(f_mid) <= tol:
                return mid
            if flo * f_mid < 0:
                hi = mid
                fhi = f_mid
            else:
                lo = mid
                flo = f_mid
        # Return the midpoint of the final bracket
        x_final = (lo + hi) / 2.0
        f_final = f(x_final)
        if abs(f_final) <= tol:
            return x_final
        # Return best_x if it's better, otherwise return x_final
        if best_f <= abs(f_final):
            return best_x
        return x_final
    
    return best_x
