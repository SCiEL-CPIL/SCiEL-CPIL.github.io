"""
Scalar root finder on 3 fixed oscillatory equations.

Strategy:
  1. Evaluate f at x0. If |f| <= 1e-8, return immediately.
  2. Use a multi-start local search to find a sign-change bracket.
     - Try a set of offsets from x0 (both directions, increasing magnitude).
     - Once a bracket [a, b] with f(a)*f(b) < 0 is found, use a
       bisection + secant hybrid (Brent-like) to converge to |f| <= 1e-12.
  3. If no bracket is found, use a damped Newton / secant local minimization
     of |f(x)| to drive f as close to zero as possible.
  4. Track the best (smallest |f|) point seen so far and return it if the
     deadline is approached.

All f calls are counted by the evaluator. The final check is not counted.
"""
import time
import math

# EVOLVE-BLOCK-START

_START_TIME = None
_DEADLINE = 140.0  # safety margin before 150 s wall-clock limit

def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME

def _is_finite(x):
    return math.isfinite(x)

def _brent_hybrid(f, a, b, fa, fb, max_iter=200):
    """
    Brent-like hybrid: maintains a bracket [a,b] and uses secant/interpolation
    when safe, bisection otherwise. Returns x with |f(x)| <= 1e-12 or best point.
    """
    # Ensure |fa| <= |fb| so we track the better endpoint
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
    
    c = a
    fc = fa
    d = b - a
    mflag = True
    eps = 1e-15
    tol = 1e-12
    
    for _ in range(max_iter):
        if _elapsed() > _DEADLINE:
            break
        if abs(fb) <= tol:
            return b
        if abs(fa) <= tol:
            return a
        
        # Check bracket validity
        if fa * fb > 0:
            break
        
        # Bisection fallback
        d = b - a
        if abs(d) < eps * max(1.0, abs(a), abs(b)):
            # Return the better endpoint
            return a if abs(fa) <= abs(fb) else b
        
        # Try secant/interpolation
        if mflag:
            # Inverse quadratic interpolation or secant
            if abs(fc) != abs(fb) and abs(fc - fa) > 1e-30:
                # Inverse quadratic
                s = (fb - fc) / (fc - fa)
                if abs(s) > 1e-15:
                    # x = b + (b - a) * (fb - fc) / (fc - fa)  -- secant
                    x_new = b + (b - a) * (fb - fc) / (fc - fa)
                    # Check if x_new is within bracket
                    if (a < x_new < b) or (b < x_new < a):
                        try:
                            fx_new = f(x_new)
                        except:
                            fx_new = None
                        if fx_new is not None and _is_finite(fx_new):
                            if abs(fx_new) <= tol:
                                return x_new
                            c = b
                            fc = fb
                            b = x_new
                            fb = fx_new
                            if abs(fb) <= abs(fa):
                                a, b = b, a
                                fa, fb = fb, fa
                            continue
            # Secant fallback
            if abs(fa - fb) > 1e-30:
                x_new = b - fb * (b - a) / (fb - fa)
                if (a < x_new < b) or (b < x_new < a):
                    try:
                        fx_new = f(x_new)
                    except:
                        fx_new = None
                    if fx_new is not None and _is_finite(fx_new):
                        if abs(fx_new) <= tol:
                            return x_new
                        c = b
                        fc = fb
                        b = x_new
                        fb = fx_new
                        if abs(fb) <= abs(fa):
                            a, b = b, a
                            fa, fb = fb, fa
                        continue
        
        # Bisection
        mid = (a + b) / 2.0
        try:
            fmid = f(mid)
        except:
            fmid = None
        
        if fmid is None or not _is_finite(fmid):
            # Can't evaluate midpoint; shrink bracket
            b = mid
            fb = fb  # keep old fb, we'll re-evaluate next iter if needed
            continue
        
        if abs(fmid) <= tol:
            return mid
        
        if fa * fmid <= 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        
        if abs(fa) <= abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        mflag = True
    
    # Return best endpoint
    return a if abs(fa) <= abs(fb) else b

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Returns a float x.
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
    
    # Phase 1: Find a sign-change bracket
    # Try offsets from x0 in increasing magnitude
    a = x0
    fa = best_f
    if fa is None or not _is_finite(fa):
        # Find a valid starting point
        for dx in [1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.1, -0.1]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    a = x0 + dx
                    fa = fx_try
                    if abs(fa) <= 1e-8:
                        return a
                    break
            except:
                continue
    
    if fa is None or not _is_finite(fa):
        return x0
    
    if abs(fa) <= 1e-8:
        return a
    
    b = None
    fb = None
    
    # Try nearby points first
    offsets = [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0,
               10.0, -10.0, 20.0, -20.0, 50.0, -50.0]
    
    for dx in offsets:
        if _elapsed() > _DEADLINE:
            break
        b_try = a + dx
        try:
            fb_try = f(b_try)
        except:
            continue
        if not _is_finite(fb_try):
            continue
        if abs(fb_try) <= 1e-8:
            return b_try
        if fa * fb_try < 0:
            b = b_try
            fb = fb_try
            break
    
    # If no bracket found with fixed offsets, try geometric expansion
    if b is None:
        for scale in [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0]:
            if _elapsed() > _DEADLINE:
                break
            for sign in [1, -1]:
                b_try = x0 + sign * scale * max(1.0, abs(x0))
                try:
                    fb_try = f(b_try)
                except:
                    continue
                if not _is_finite(fb_try):
                    continue
                if abs(fb_try) <= 1e-8:
                    return b_try
                if fa * fb_try < 0:
                    b = b_try
                    fb = fb_try
                    break
            if b is not None:
                break
    
    # Phase 2: If bracket found, use Brent hybrid
    if b is not None and fb is not None and fa * fb < 0:
        result = _brent_hybrid(f, a, b, fa, fb)
        try:
            fr = f(result)
        except:
            fr = float('inf')
        if _is_finite(fr) and abs(fr) <= 1e-8:
            return result
        # Update best
        if _is_finite(fr) and (best_f is None or not _is_finite(best_f) or abs(fr) < abs(best_f)):
            best_x = result
            best_f = fr
    
    # Phase 3: No bracket found (or bracket didn't converge well enough).
    # Use damped Newton / secant to minimize |f(x)| locally.
    x = x0
    fx = best_f
    if fx is None or not _is_finite(fx):
        for dx in [0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
            try:
                fx_try = f(x0 + dx)
                if _is_finite(fx_try):
                    x = x0 + dx
                    fx = fx_try
                    break
            except:
                continue
    
    if fx is None or not _is_finite(fx):
        return best_x
    
    # Track previous point for secant
    x_prev = x + 1e-4
    try:
        f_prev = f(x_prev)
    except:
        f_prev = None
    
    for _ in range(100):
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
                    # Damped step
                    step = x_new - x
                    if abs(step) > max(1.0, abs(x)):
                        x_new = x + 0.5 * step * (1 if step > 0 else -1)
                    
                    try:
                        fx_new = f(x_new)
                    except:
                        fx_new = None
                    
                    if fx_new is not None and _is_finite(fx_new):
                        if abs(fx_new) <= 1e-8:
                            return x_new
                        if abs(fx_new) < abs(fx):
                            x_prev = x
                            f_prev = fx
                            x = x_new
                            fx = fx_new
                        else:
                            # Try smaller step
                            x_new2 = x - 0.1 * fx / deriv
                            if _is_finite(x_new2):
                                try:
                                    fx_new2 = f(x_new2)
                                except:
                                    fx_new2 = None
                                if fx_new2 is not None and _is_finite(fx_new2):
                                    if abs(fx_new2) <= 1e-8:
                                        return x_new2
                                    if abs(fx_new2) < abs(fx):
                                        x_prev = x
                                        f_prev = fx
                                        x = x_new2
                                        fx = fx_new2
                    # Update best
                    if (best_f is None or not _is_finite(best_f) or abs(fx) < abs(best_f)):
                        best_x = x
                        best_f = fx
        else:
            break
    
    # Update best one more time
    if _is_finite(fx) and (best_f is None or not _is_finite(best_f) or abs(fx) < abs(best_f)):
        best_x = x
        best_f = fx
    
    return best_x

# EVOLVE-BLOCK-END
