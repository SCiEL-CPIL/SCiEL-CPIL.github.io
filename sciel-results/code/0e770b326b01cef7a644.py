"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 120.0
_SAFETY_MARGIN = 5.0

def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)

def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    """
    Safeguarded Inverse Quadratic Interpolation (IQI) root finder with caching.
    Strategy:
    1. Check x0.
    2. Try to find a local bracket around x0 by expanding steps.
    3. If bracket found, use Hybrid IQI/Bisection (Brent-like but optimized for calls).
    4. If no bracket found, use Secant method with local perturbations (IQI-like if 3 points available).
    5. Cache all evaluations to minimize call count.
    """
    x0 = float(x0)
    cache = {}
    call_count = 0
    tol = 1e-8
    tol_sq = tol * tol
    
    def f_cached(x):
        nonlocal call_count
        # Use a key that distinguishes very close floats if needed, but usually exact match is fine
        # For floats, direct hash is okay for our purposes in this context (no symbolic math)
        key = x
        if key in cache:
            return cache[key]
        
        try:
            val = f(x)
            call_count += 1
            if _is_finite(val):
                cache[key] = val
                return val
            else:
                # Store non-finite to avoid re-evaluating, but mark as bad
                cache[key] = float('nan')
                return float('nan')
        except Exception:
            cache[key] = float('nan')
            return float('nan')

    # 1. Check x0
    fx0 = f_cached(x0)
    if _is_finite(fx0) and abs(fx0) <= tol:
        return x0

    if not _is_finite(fx0):
        # If x0 is invalid, try a small perturbation
        x_try = x0 + 1e-6
        fx_try = f_cached(x_try)
        if _is_finite(fx_try) and abs(fx_try) <= tol:
            return x_try
        if not _is_finite(fx_try):
            return x0 # Return best effort

    # 2. Try to find a bracket
    # Expand from x0 in both directions
    step = 1e-4
    max_expansions = 30
    a, b = None, None
    fa, fb = None, None
    
    # We already have x0, fx0. Let's call it c for now.
    c = x0
    fc = fx0

    for i in range(max_expansions):
        if _time_left() < 1.0:
            break
            
        # Try right
        x_r = c + step
        fx_r = f_cached(x_r)
        if _is_finite(fx_r):
            if fc * fx_r < 0:
                a, b = c, x_r
                fa, fb = fc, fx_r
                break
            # If fx_r is close to 0, return it
            if abs(fx_r) <= tol:
                return x_r
        
        # Try left
        x_l = c - step
        fx_l = f_cached(x_l)
        if _is_finite(fx_l):
            if fc * fx_l < 0:
                a, b = x_l, c
                fa, fb = fx_l, fc
                break
            if abs(fx_l) <= tol:
                return x_l
        
        step *= 2.0
        if step > 1e6:
            break

    # 3. If bracket found, use Hybrid IQI/Bisection
    if a is not None and b is not None:
        # Ensure fa corresponds to a and fb to b
        # We want |fa| < |fb| so that 'b' is the best estimate
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        # a, b is the bracket, b is the best guess
        for _ in range(50):
            if _time_left() < 0.5:
                break
            if abs(fb) <= tol:
                return b
            if abs(b - a) < 1e-14:
                break
                
            # Inverse Quadratic Interpolation
            # We need a third point. We have a and b. 
            # We can use the previous 'b' (which is now 'a') or just use Secant if we don't have a distinct 3rd point.
            # Actually, standard IQI needs 3 points: x_{n-2}, x_{n-1}, x_n.
            # Here we have a bracket [a, b]. 
            # Let's use the point with the smaller |f| as the "best" and the other as "second".
            # To do IQI, we need a third point. We can generate one or use a Secant step if we don't have history.
            # For simplicity and robustness in a bracket, let's use a safeguarded Secant step first, 
            # and if we have a third point from previous iterations, use IQI.
            
            # Let's implement a simple IQI step using a, b, and a third point 'c'.
            # We don't have a 'c' yet in this loop scope easily unless we track history.
            # Let's use a Secant step from a and b, but safeguarded by bisection.
            # Secant: x_new = b - fb * (b - a) / (fb - fa)
            
            denom = fb - fa
            if denom == 0:
                x_new = (a + b) / 2.0
            else:
                x_new = b - fb * (b - a) / denom
            
            # Safeguard: if x_new is not in (a, b), use bisection
            if not (a < x_new < b):
                x_new = (a + b) / 2.0
            
            fx_new = f_cached(x_new)
            if not _is_finite(fx_new):
                # If new point is bad, bisect
                x_new = (a + b) / 2.0
                fx_new = f_cached(x_new)
                if not _is_finite(fx_new):
                    break

            if abs(fx_new) <= tol:
                return x_new

            # Update bracket
            if fa * fx_new < 0:
                b = x_new
                fb = fx_new
            else:
                a = x_new
                fa = fx_new
            
            # Ensure b is the best estimate
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

        return b if abs(fb) <= tol else a

    # 4. No bracket found. Use Secant/IQI without bracket (unbounded)
    # We have x0, fx0. Let's pick a second point.
    # Try a small perturbation
    h = 1e-6 * max(1.0, abs(x0))
    x1 = x0 + h
    fx1 = f_cached(x1)
    
    if not _is_finite(fx1):
        # Try other direction
        x1 = x0 - h
        fx1 = f_cached(x1)
        
    if not _is_finite(fx1):
        # If we can't even evaluate a neighbor, return x0
        return x0

    if abs(fx1) <= tol:
        return x1

    # Secant loop
    x_prev = x0
    x_curr = x1
    f_prev = fx0
    f_curr = fx1
    
    for _ in range(50):
        if _time_left() < 0.5:
            break
        if abs(f_curr) <= tol:
            return x_curr
            
        denom = f_curr - f_prev
        if denom == 0:
            break
            
        x_new = x_curr - f_curr * (x_curr - x_prev) / denom
        
        if not _is_finite(x_new):
            break
            
        f_new = f_cached(x_new)
        if not _is_finite(f_new):
            break
            
        if abs(f_new) <= tol:
            return x_new
            
        # Update
        x_prev = x_curr
        f_prev = f_curr
        x_curr = x_new
        f_curr = f_new
        
        # Divergence check
        if abs(x_curr - x_prev) < 1e-14:
            break

    return x_curr if abs(f_curr) <= tol else x0
# EVOLVE-BLOCK-END
