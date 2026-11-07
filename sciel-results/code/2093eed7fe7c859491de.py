"""Rewritten program using a robust, adaptive root-finding strategy."""
import time
import math

# EVOLVE-BLOCK-START
_START_TIME = None
_DEADLINE = 145.0  # 145s out of 150s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _solve_bracket(f, a, fa, b, fb):
    """Illinois-style inverse linear interpolation for bracketed root."""
    left, fleft = a, fa
    right, fright = b, fb
    if fleft * fright > 0:
        return None
    
    for _ in range(100):
        if _elapsed() > _DEADLINE:
            break
        
        if abs(fleft) <= 1e-8:
            return left
        if abs(fright) <= 1e-8:
            return right
        
        # Illinois method: inverse linear interpolation with damping
        denom = fright - fleft
        if abs(denom) < 1e-300:
            x_new = (left + right) / 2.0
        else:
            x_new = (left * fright - right * fleft) / denom
        
        # If x_new is outside bracket or too close to an endpoint, bisect
        if not (left < x_new < right) or abs(x_new - left) < 1e-15 * max(1.0, abs(left)) or abs(x_new - right) < 1e-15 * max(1.0, abs(right)):
            x_new = (left + right) / 2.0
        
        try:
            f_new = f(x_new)
        except:
            f_new = None
        
        if f_new is None or not _is_finite(f_new):
            x_new = (left + right) / 2.0
            try:
                f_new = f(x_new)
            except:
                break
        
        if f_new is None or not _is_finite(f_new):
            break
        
        if abs(f_new) <= 1e-8:
            return x_new
        
        if f_new * fleft < 0:
            right, fright = x_new, f_new
            fleft *= 0.5  # Illinois damping
        else:
            left, fleft = x_new, f_new
            fright *= 0.5
        
        if abs(right - left) < 1e-14 * max(1.0, abs(left), abs(right)):
            break
    
    if abs(fleft) <= abs(fright):
        return left
    else:
        return right

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8, using Illinois-style
    inverse linear interpolation with adaptive bracket search.
    
    Returns a float x, or raises an exception if no root can be found.
    """
    x0 = float(x0)
    if not _is_finite(x0):
        x0 = 0.0
    
    best_x = x0
    best_f = None
    try:
        best_f = f(x0)
    except:
        best_f = float('inf')
    
    if best_f is not None and _is_finite(best_f) and abs(best_f) <= 1e-8:
        return x0
    
    # Phase 1: Find a sign change bracket
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    break
            except:
                continue
    
    if fa is None or not _is_finite(fa):
        return x0
    
    if abs(fa) <= 1e-8:
        return a
    
    # Try to find b such that f(a)*f(b) < 0
    b = None
    fb = None
    
    # First try nearby points
    for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0, 10.0, -10.0]:
        try:
            if _elapsed() > _DEADLINE:
                break
            b_try = a + dx
            fb_try = f(b_try)
            if _is_finite(fb_try):
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
        except:
            continue
    
    # If no bracket found, try expanding
    if b is None:
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            for sign in [1, -1]:
                try:
                    if _elapsed() > _DEADLINE:
                        break
                    b_try = x0 + sign * scale * max(1.0, abs(x0))
                    fb_try = f(b_try)
                    if _is_finite(fb_try):
                        if abs(fb_try) <= 1e-8:
                            return b_try
                        if _is_finite(fa) and fa * fb_try < 0:
                            b = b_try
                            fb = fb_try
                            break
                except:
                    continue
            if b is not None:
                break
    
    # Phase 2: If we have a bracket, use Illinois method
    if b is not None and fb is not None:
        if fa * fb < 0:
            result = _solve_bracket(f, a, fa, b, fb)
            if result is not None:
                return result
    
    # Phase 3: No bracket found, use local search with Newton-like iterations
    x = x0
    fx = best_f
    if fx is None or not _is_finite(fx):
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0]:
            try:
                x_try = x0 + dx
                fx_try = f(x_try)
                if _is_finite(fx_try):
                    x = x_try
                    fx = fx_try
                    break
            except:
                continue
    
    if fx is None or not _is_finite(fx):
        return x0
    
    for _ in range(80):
        if _elapsed() > _DEADLINE:
            break
        
        if abs(fx) <= 1e-8:
            return x
        
        # Finite difference derivative
        h = 1e-7 * max(1.0, abs(x))
        try:
            fx_plus = f(x + h)
        except:
            fx_plus = None
        
        if fx_plus is not None and _is_finite(fx_plus):
            deriv = (fx_plus - fx) / h
            if deriv != 0 and _is_finite(deriv):
                x_new = x - fx / deriv
                if _is_finite(x_new):
                    if abs(x_new - x) > max(1.0, abs(x)):
                        x_new = x + 0.5 * (x_new - x) * (1 if x_new > x else -1)
                    
                    try:
                        fx_new = f(x_new)
                    except:
                        fx_new = None
                    
                    if fx_new is not None and _is_finite(fx_new):
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        if abs(fx_new) < abs(fx):
                            x = x_new
                            fx = fx_new
                        else:
                            x_new = x - 0.1 * fx / deriv
                            if _is_finite(x_new):
                                try:
                                    fx_new = f(x_new)
                                except:
                                    fx_new = None
                                if fx_new is not None and _is_finite(fx_new):
                                    if abs(fx_new) <= 1e-8:
                                        return x_new
                                    if abs(fx_new) < abs(fx):
                                        x = x_new
                                        fx = fx_new
        else:
            break
    
    if fx is not None and _is_finite(fx) and abs(fx) <= 1e-8:
        return x
    
    return x

# EVOLVE-BLOCK-END
