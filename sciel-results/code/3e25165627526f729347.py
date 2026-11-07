"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
import numpy as np

def _brentq(f, a, b, fa, fb, maxiter=100):
    """
    Brent's method with initial values fa, fb.
    Returns x such that f(x) is close to 0.
    """
    if fa == 0.0:
        return a
    if fb == 0.0:
        return b
    
    # Ensure fa and fb have opposite signs
    if fa * fb > 0:
        # No sign change, return the one with smaller |f|
        return a if abs(fa) < abs(fb) else b

    c = a
    fc = fa
    d = b - a
    e = d
    
    for i in range(maxiter):
        if fb * fc > 0:
            c = a
            fc = fa
            d = e = b - a
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        tol = 2.2e-16 * abs(b) + 1e-14
        m = 0.5 * (c - b)
        
        if abs(m) < tol or fb == 0.0:
            return b
        
        if abs(e) < tol or abs(fa) <= abs(fb):
            # Bisection
            d = m
            e = d
        else:
            # Inverse quadratic interpolation
            S = fb / fa
            A = fc / fa
            B = A * (A - S) - (b - c) / (c - a) * (S - 1.0)
            
            if a == b:
                p = A * 2.0 * m
                q = 1.0 - A
            else:
                p = A * (2.0 * m * (b - c) * (fc - fb) - (b - a) * (fc - fa))
                q = (fc - fb) * (fb - fa) * (fc - fa)
            
            if 2.0 * p > q * min(1.0, 2.0 * m / abs(m) if m != 0 else 1.0):
                p = -q
            else:
                p = 2.0 * p
            
            if 2.0 * p > abs(q * e):
                d = m
                e = d
            else:
                e = d
                d = p / q
        
        a = b
        fa = fb
        b = b + d
        fb = f(b)
        
        if not np.isfinite(fb):
            # Fallback to bisection if we hit a non-finite value
            b = 0.5 * (a + b)
            fb = f(b)
            if not np.isfinite(fb):
                break

    return b

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    x = float(x0)
    
    # Phase 1: Try Newton's method with a few iterations
    # This is fast if it works.
    for _ in range(10):
        if time.time() > deadline:
            break
        fx = f(x)
        if not np.isfinite(fx):
            break
        if abs(fx) <= 1e-8:
            return x
            
        # Central difference for better derivative approximation
        # But that costs 2 calls. Let's use forward difference to save calls.
        h = 1e-5 * max(1.0, abs(x))
        fx_plus = f(x + h)
        if not np.isfinite(fx_plus):
            break
            
        d = (fx_plus - fx) / h
        if d == 0 or not np.isfinite(d):
            break
            
        x_new = x - fx / d
        if not np.isfinite(x_new):
            break
            
        # Check for oscillation or divergence
        if abs(x_new - x) > 1e6 * max(1.0, abs(x)):
            break
            
        x = x_new

    # Phase 2: If Newton failed or didn't converge, use a robust bracketing method
    # First, try to find a bracket around x0
    # We'll search exponentially in both directions
    
    def try_brent_bracket(x0, fx0):
        # Try small steps first
        for step in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]:
            if time.time() > deadline:
                return None
                
            a = x0 - step
            b = x0 + step
            
            fa = f(a)
            if not np.isfinite(fa):
                continue
            fb = f(b)
            if not np.isfinite(fb):
                continue
            
            if fa * fb < 0:
                # Sign change found
                return _brentq(f, a, b, fa, fb)
            
            # If no sign change, check if we are close
            if abs(fa) < abs(fx0) or abs(fb) < abs(fx0):
                # Move x0 to the better point and try again with larger step?
                # Or just continue searching
                pass
        
        return None
    
    # Get current fx
    fx = f(x) if not np.isfinite(f(x)) else f(x) # Ensure we have a value
    # Actually, we might have lost track of fx if we broke out of Newton loop.
    # Let's just re-evaluate or use the last known.
    # To be safe, let's just start bracketing from x0.
    
    result = try_brent_bracket(x0, f(x0))
    if result is not None and np.isfinite(result):
        return result
    
    # If bracketing failed, try a global grid search to find a sign change
    # This is expensive but robust for oscillatory functions.
    
    # Determine a reasonable domain
    # x0 is the starting point. Roots are often near x0 or within a few units.
    # For oscillatory functions, we might need to look further.
    
    # Let's try a grid search in [-100, 100] or around x0
    # But we don't know the scale.
    
    # Strategy: Exponential search for sign change from x0
    # We already did this in try_brent_bracket.
    
    # If that failed, maybe the root is not a sign change (tangent root)?
    # Or the function is not continuous?
    
    # Let's try a finer local search around x0 if we haven't already.
    # Or use scipy.optimize.brentq with a wider bracket.
    
    # Let's try a few specific brackets
    brackets = [
        (x0 - 1e-6, x0 + 1e-6),
        (x0 - 1e-5, x0 + 1e-5),
        (x0 - 1e-4, x0 + 1e-4),
        (x0 - 1e-3, x0 + 1e-3),
        (x0 - 1e-2, x0 + 1e-2),
        (x0 - 0.1, x0 + 0.1),
        (x0 - 1.0, x0 + 1.0),
        (x0 - 10.0, x0 + 10.0),
        (x0 - 100.0, x0 + 100.0),
        (x0 - 1000.0, x0 + 1000.0),
    ]
    
    for a, b in brackets:
        if time.time() > deadline:
            break
        fa = f(a)
        if not np.isfinite(fa):
            continue
        fb = f(b)
        if not np.isfinite(fb):
            continue
        if fa * fb < 0:
            res = _brentq(f, a, b, fa, fb)
            if np.isfinite(res):
                return res
    
    # If all else fails, return the best x we have
    # We should probably evaluate f(x) one last time to be sure, but we are out of calls.
    # Just return x.
    return x

# EVOLVE-BLOCK-END
