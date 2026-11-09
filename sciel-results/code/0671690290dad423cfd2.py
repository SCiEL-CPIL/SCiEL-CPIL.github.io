"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

_T0 = time.time()
_LIMIT = 110.0

def _left():
    return _LIMIT - (time.time() - _T0)

def _fin(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)

def find_root(f, x0):
    x0 = float(x0)
    best_x = x0
    best_fx = float('inf')
    
    # Evaluate starting point
    try:
        fx0 = f(x0)
        if _fin(fx0):
            best_fx = fx0
            best_x = x0
            if abs(fx0) <= 1e-8:
                return x0
    except Exception:
        pass
    
    # Strategy 1: Newton's method with adaptive step and damping
    x = x0
    for i in range(40):
        if _left() < 2.0:
            break
        try:
            fx = f(x)
        except Exception:
            break
        if not _fin(fx):
            break
        if abs(fx) < abs(best_fx):
            best_fx, best_x = fx, x
        if abs(fx) <= 1e-8:
            return x
        
        # Finite difference derivative with adaptive h
        h = 1e-5 * max(1.0, abs(x))
        try:
            fxh = f(x + h)
        except Exception:
            break
        if not _fin(fxh):
            # Try smaller h
            h = 1e-7 * max(1.0, abs(x))
            try:
                fxh = f(x + h)
            except Exception:
                break
            if not _fin(fxh):
                break
        d = (fxh - fx) / h
        if not _fin(d) or abs(d) < 1e-15:
            break
        
        # Newton step with damping if step is too large
        step = fx / d
        if abs(step) > 10.0 * max(1.0, abs(x)):
            step = math.copysign(10.0 * max(1.0, abs(x)), step)
        x_new = x - step
        if not _fin(x_new):
            break
        
        # Check if new point is better
        try:
            fxn = f(x_new)
        except Exception:
            break
        if not _fin(fxn):
            # Damp the step further
            x_new = x - 0.5 * step
            if not _fin(x_new):
                break
            try:
                fxn = f(x_new)
            except Exception:
                break
            if not _fin(fxn):
                break
        
        if abs(fxn) < abs(best_fx):
            best_fx, best_x = fxn, x_new
        if abs(fxn) <= 1e-8:
            return x_new
        
        # If the step didn't help, damp it
        if abs(fxn) > abs(fx):
            x_new = x - 0.5 * step
            if not _fin(x_new):
                break
            try:
                fxn = f(x_new)
            except Exception:
                break
            if not _fin(fxn):
                break
            if abs(fxn) < abs(best_fx):
                best_fx, best_x = fxn, x_new
            if abs(fxn) <= 1e-8:
                return x_new
            if abs(fxn) > abs(fx):
                break
        
        x = x_new
    
    # Strategy 2: Secant method
    if _left() > 2.0 and abs(best_fx) > 1e-8:
        x1 = x0
        x2 = x0 + 0.5 * max(1.0, abs(x0))
        try:
            f1 = f(x1)
        except Exception:
            f1 = None
        try:
            f2 = f(x2)
        except Exception:
            f2 = None
        
        if f1 is not None and _fin(f1):
            if abs(f1) < abs(best_fx):
                best_fx, best_x = f1, x1
            if abs(f1) <= 1e-8:
                return x1
        if f2 is not None and _fin(f2):
            if abs(f2) < abs(best_fx):
                best_fx, best_x = f2, x2
            if abs(f2) <= 1e-8:
                return x2
        
        if f1 is not None and f2 is not None and _fin(f1) and _fin(f2) and f2 != f1:
            for i in range(30):
                if _left() < 2.0:
                    break
                x_new = x2 - f2 * (x2 - x1) / (f2 - f1)
                if not _fin(x_new):
                    break
                try:
                    f_new = f(x_new)
                except Exception:
                    break
                if not _fin(f_new):
                    break
                if abs(f_new) < abs(best_fx):
                    best_fx, best_x = f_new, x_new
                if abs(f_new) <= 1e-8:
                    return x_new
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
    
    # Strategy 3: Bisection with bracket expansion
    if _left() > 2.0 and abs(best_fx) > 1e-8:
        # Try to find a bracket around x0
        for scale in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
            if _left() < 2.0:
                break
            a = x0 - scale
            b = x0 + scale
            try:
                fa = f(a)
            except Exception:
                fa = None
            try:
                fb = f(b)
            except Exception:
                fb = None
            
            if fa is not None and _fin(fa):
                if abs(fa) < abs(best_fx):
                    best_fx, best_x = fa, a
                if abs(fa) <= 1e-8:
                    return a
            if fb is not None and _fin(fb):
                if abs(fb) < abs(best_fx):
                    best_fx, best_x = fb, b
                if abs(fb) <= 1e-8:
                    return b
            
            if fa is not None and fb is not None and _fin(fa) and _fin(fb) and fa * fb < 0:
                # Bisection
                for i in range(60):
                    if _left() < 1.0:
                        break
                    mid = (a + b) / 2.0
                    try:
                        fm = f(mid)
                    except Exception:
                        break
                    if not _fin(fm):
                        break
                    if abs(fm) < abs(best_fx):
                        best_fx, best_x = fm, mid
                    if abs(fm) <= 1e-8:
                        return mid
                    if fa * fm < 0:
                        b, fb = mid, fm
                    else:
                        a, fa = mid, fm
                    if abs(b - a) < 1e-14:
                        break
                break
    
    # Strategy 4: Local minimization of |f(x)| using a simple gradient-free approach
    if _left() > 2.0 and abs(best_fx) > 1e-8:
        x = best_x
        step = 0.1 * max(1.0, abs(x))
        for i in range(20):
            if _left() < 1.0:
                break
            best_local = (x, abs(best_fx))
            for dx in [-step, step, -0.5*step, 0.5*step, -2*step, 2*step]:
                x_try = x + dx
                if not _fin(x_try):
                    continue
                try:
                    fx_try = f(x_try)
                except Exception:
                    continue
                if not _fin(fx_try):
                    continue
                if abs(fx_try) < best_local[1]:
                    best_local = (x_try, abs(fx_try))
                    if abs(fx_try) < abs(best_fx):
                        best_fx, best_x = fx_try, x_try
                    if abs(fx_try) <= 1e-8:
                        return x_try
            if best_local[0] == x:
                step *= 0.5
                if step < 1e-10:
                    break
            else:
                x = best_local[0]
    
    # Strategy 5: High-precision polish pass using Newton with tighter tolerance
    if _left() > 1.0 and abs(best_fx) > 1e-8:
        x = best_x
        for i in range(30):
            if _left() < 0.5:
                break
            try:
                fx = f(x)
            except Exception:
                break
            if not _fin(fx):
                break
            if abs(fx) < abs(best_fx):
                best_fx, best_x = fx, x
            if abs(fx) <= 1e-8:
                return x
            
            # Use higher-order finite difference for better derivative estimate
            h = 1e-6 * max(1.0, abs(x))
            try:
                fx_h = f(x + h)
                fx_hm = f(x - h)
            except Exception:
                break
            if not _fin(fx_h) or not _fin(fx_hm):
                break
            
            # Central difference
            d = (fx_h - fx_hm) / (2.0 * h)
            if not _fin(d) or abs(d) < 1e-14:
                break
            
            step = fx / d
            # Limit step size
            max_step = 1.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
            
            x_new = x - step
            if not _fin(x_new):
                break
            
            try:
                fxn = f(x_new)
            except Exception:
                break
            if not _fin(fxn):
                break
            
            if abs(fxn) < abs(best_fx):
                best_fx, best_x = fxn, x_new
            if abs(fxn) <= 1e-8:
                return x_new
            
            # If step made things worse, damp
            if abs(fxn) > abs(fx):
                x_new = x - 0.5 * step
                if not _fin(x_new):
                    break
                try:
                    fxn = f(x_new)
                except Exception:
                    break
                if not _fin(fxn):
                    break
                if abs(fxn) < abs(best_fx):
                    best_fx, best_x = fxn, x_new
                if abs(fxn) <= 1e-8:
                    return x_new
                if abs(fxn) > abs(fx):
                    break
            
            x = x_new
    
    return best_x
# EVOLVE-BLOCK-END
