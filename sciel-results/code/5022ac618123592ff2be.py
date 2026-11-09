import math
import time

def find_root(f, x0):
    tol = 1e-8
    deadline = time.monotonic() + 110.0  # safety margin before 120s limit
    
    x = float(x0)
    fx = f(x)
    
    if abs(fx) <= tol:
        return x
    
    # Try to find a bracket [a, b] where f(a) and f(b) have opposite signs
    # Use geometric expansion from x0
    a = x
    fa = fx
    
    # Try small steps first to catch nearby roots
    deltas = [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]
    
    bracket_found = False
    b = None
    fb = None
    
    for delta in deltas:
        if time.monotonic() > deadline:
            break
        
        # Check x + delta
        x_plus = x + delta
        f_plus = f(x_plus)
        if abs(f_plus) <= tol:
            return x_plus
        if fa * f_plus < 0:
            a, fa = x, fa
            b, fb = x_plus, f_plus
            bracket_found = True
            break
        
        # Check x - delta
        x_minus = x - delta
        f_minus = f(x_minus)
        if abs(f_minus) <= tol:
            return x_minus
        if fa * f_minus < 0:
            a, fa = x_minus, f_minus
            b, fb = x, fa
            bracket_found = True
            break
    
    if not bracket_found:
        # Fallback: try a wider search with more aggressive expansion
        # Expand in both directions with larger steps
        step = 1e2
        for _ in range(20):
            if time.monotonic() > deadline:
                break
            step *= 2
            x_right = x + step
            f_right = f(x_right)
            if abs(f_right) <= tol:
                return x_right
            if fa * f_right < 0:
                b, fb = x_right, f_right
                bracket_found = True
                break
            
            x_left = x - step
            f_left = f(x_left)
            if abs(f_left) <= tol:
                return x_left
            if fa * f_left < 0:
                a, fa = x_left, f_left
                b, fb = x, fx
                bracket_found = True
                break
        
        if not bracket_found:
            # Return best guess - the point with smallest |f|
            # We don't have a great candidate, return x0
            return x
    
    # Bisection method on [a, b]
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa
    
    # Bisection: converges in ~log2((b-a)/tol) iterations
    # For a bracket of width W, need ~log2(W/1e-8) iterations
    # Cap iterations to avoid infinite loops
    max_iter = 100  # 2^100 is way more than enough for any reasonable bracket
    
    for _ in range(max_iter):
        if time.monotonic() > deadline:
            break
        
        mid = (a + b) / 2.0
        fmid = f(mid)
        
        if abs(fmid) <= tol:
            return mid
        
        if fa * fmid <= 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        
        # Check if bracket is too small
        if abs(b - a) < 1e-14:
            break
    
    # Return the midpoint of the final bracket
    return (a + b) / 2.0
