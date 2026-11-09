"""Hybrid root-finder: Newton with secant fallback and Brentq bracketing."""
import math
import time

# Deadline guard: kill after 150s wall-clock, use 145s as safety margin
_DEADLINE = 145.0

def _elapsed():
    return time.time() - _start_time

_start_time = time.time()

def find_root(f, x0):
    global _start_time
    _start_time = time.time()
    deadline = _start_time + _DEADLINE
    x0 = float(x0)
    
    # We'll track the best candidate found so far
    best_x = x0
    best_fx = None
    try:
        best_fx = f(x0)
    except:
        pass
    
    def _update_best(x, fx):
        nonlocal best_x, best_fx
        if fx is not None and math.isfinite(fx):
            if best_fx is None or abs(fx) < abs(best_fx):
                best_x = x
                best_fx = fx
    
    def _check_deadline():
        return time.time() > deadline - 0.5
    
    # Step 1: Try to bracket a root using adaptive sampling
    # Start with a small window and expand if needed
    def _try_bracket(x0, max_expansions=10):
        """Try to find a bracket [a, b] where f(a)*f(b) < 0."""
        h = 1e-4 * max(1.0, abs(x0))
        for expansion in range(max_expansions):
            a = x0 - h
            b = x0 + h
            try:
                fa = f(a)
                fb = f(b)
            except:
                h *= 2
                continue
            
            if not (math.isfinite(fa) and math.isfinite(fb)):
                h *= 2
                continue
            
            _update_best(a, fa)
            _update_best(b, fb)
            
            if _check_deadline():
                return None
            
            if fa * fb < 0:
                return (a, b, fa, fb)
            
            # Check if either endpoint is already a good root
            if abs(fa) <= 1e-8:
                return (a, a, fa, fb)
            if abs(fb) <= 1e-8:
                return (b, b, fa, fb)
            
            h *= 2.0
        
        return None
    
    # Step 2: Newton's method with secant fallback
    def _newton_secant(x0, max_iter=50):
        """Newton with secant fallback. Returns (x, fx, converged)."""
        x = float(x0)
        try:
            fx = f(x)
        except:
            return (x, fx, False)
        
        _update_best(x, fx)
        
        if abs(fx) <= 1e-12:
            return (x, fx, True)
        
        # Secant method state
        x_prev = x - 1e-6 * max(1.0, abs(x))
        try:
            fx_prev = f(x_prev)
        except:
            fx_prev = None
        
        _update_best(x_prev, fx_prev)
        
        for i in range(max_iter):
            if _check_deadline():
                return (x, fx, False)
            
            if abs(fx) <= 1e-12:
                return (x, fx, True)
            
            # Try Newton step using finite difference derivative
            h = 1e-7 * max(1.0, abs(x))
            try:
                fx_plus = f(x + h)
            except:
                fx_plus = None
            
            if fx_plus is not None and math.isfinite(fx_plus):
                d = (fx_plus - fx) / h
                if d != 0 and math.isfinite(d):
                    x_new = x - fx / d
                    if math.isfinite(x_new):
                        try:
                            fx_new = f(x_new)
                        except:
                            fx_new = None
                        
                        if fx_new is not None and math.isfinite(fx_new):
                            _update_best(x_new, fx_new)
                            if abs(fx_new) <= 1e-12:
                                return (x_new, fx_new, True)
                            
                            # Accept Newton step if it reduces |f|
                            if abs(fx_new) < abs(fx) * 1.5:
                                x, fx = x_new, fx_new
                                continue
                        else:
                            # f(x_new) failed, fall through to secant
                            pass
            
            # Secant step
            if fx_prev is not None and math.isfinite(fx_prev) and fx_prev != fx:
                d_sec = (fx - fx_prev) / (x - x_prev)
                if d_sec != 0 and math.isfinite(d_sec):
                    x_new = x - fx / d_sec
                    if math.isfinite(x_new):
                        try:
                            fx_new = f(x_new)
                        except:
                            fx_new = None
                        
                        if fx_new is not None and math.isfinite(fx_new):
                            _update_best(x_new, fx_new)
                            if abs(fx_new) <= 1e-12:
                                return (x_new, fx_new, True)
                            
                            # Shift secant window
                            x_prev, fx_prev = x, fx
                            x, fx = x_new, fx_new
                            continue
            
            # If neither Newton nor secant worked, try a small perturbation
            x = x + 1e-6 * max(1.0, abs(x))
            try:
                fx = f(x)
            except:
                fx = None
            _update_best(x, fx)
            if fx is None or not math.isfinite(fx):
                break
        
        return (x, fx, abs(fx) is not None and math.isfinite(fx) and abs(fx) <= 1e-8)
    
    # Step 3: Brent's method for bracketed roots
    def _brent(a, b, fa, fb, max_iter=100):
        """Brent's method given a bracket [a, b] with f(a)*f(b) < 0."""
        # We'll implement a simplified Brent's method
        # Using inverse quadratic interpolation with bisection fallback
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        for i in range(max_iter):
            if _check_deadline():
                return (b, fb, False)
            
            if abs(fb) <= 1e-12:
                return (b, fb, True)
            if abs(fc) <= 1e-12:
                return (c, fc, True)
            
            if fa * fb > 0:
                # Shouldn't happen if bracket is valid, but handle it
                a, b, c = c, a, b
                fa, fb, fc = fc, fa, fb
            
            # Check if b is close enough
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                return (b, fb, abs(fb) <= 1e-8)
            
            # Inverse quadratic interpolation or bisection
            use_qi = False
            if e != 0 and d != 0:
                # Try inverse quadratic interpolation
                if fa == fc:
                    # Linear interpolation
                    m = 0.5 * (c - b)
                    if abs(m) >= 1e-14:
                        x_new = b - fb * m / (0.5 * (fc - fb))
                    else:
                        x_new = b
                elif fb == fc:
                    m = 0.5 * (a - b)
                    if abs(m) >= 1e-14:
                        x_new = b - fb * m / (0.5 * (fa - fb))
                    else:
                        x_new = b
                else:
                    # Inverse quadratic interpolation
                    num1 = fa * fb
                    num2 = fb * fc
                    num3 = fa * fc
                    den1 = (fa - fb) * (fb - fc)
                    den2 = (fb - fa) * (fc - fb)
                    den3 = (fc - fa) * (fa - fb)
                    
                    if den1 != 0 and den2 != 0 and den3 != 0:
                        x_new = a * num1 / (den1 * (a - b) * (a - c)) + \
                                b * num2 / (den2 * (b - a) * (b - c)) + \
                                c * num3 / (den3 * (c - a) * (c - b))
                        use_qi = True
                    else:
                        # Fall back to secant
                        if fa != fb:
                            x_new = b - fb * (b - a) / (fb - fa)
                            use_qi = True
                        else:
                            x_new = 0.5 * (a + b)
                
                if use_qi:
                    # Check if x_new is in the bracket
                    if not (min(a, b) < x_new < max(a, b)):
                        x_new = 0.5 * (a + b)
                    else:
                        # Check if x_new is close to b
                        if abs(x_new - b) >= abs(e) * 0.5:
                            e = d
                            d = x_new - b
                        else:
                            d = x_new - b
                            e = 0.5 * d
            else:
                # Bisection
                x_new = 0.5 * (a + b)
                e = d
                d = x_new - b
            
            try:
                f_new = f(x_new)
            except:
                f_new = None
            
            if f_new is None or not math.isfinite(f_new):
                # Fall back to bisection
                x_new = 0.5 * (a + b)
                try:
                    f_new = f(x_new)
                except:
                    f_new = None
            
            if f_new is None or not math.isfinite(f_new):
                break
            
            _update_best(x_new, f_new)
            
            if abs(f_new) <= 1e-12:
                return (x_new, f_new, True)
            
            # Update bracket
            if fa * f_new <= 0:
                b, fb = x_new, f_new
            else:
                a, fa = x_new, f_new
            
            # Check convergence
            if abs(b - a) < 1e-14 * max(1.0, abs(b)):
                return (b, fb, abs(fb) <= 1e-8)
        
        return (b, fb, abs(fb) is not None and math.isfinite(fb) and abs(fb) <= 1e-8)
    
    # Main strategy:
    # 1. First try Newton/secant from x0
    # 2. If that fails or doesn't converge, try to bracket and use Brent
    
    # Try Newton/secant first
    x_newton, fx_newton, converged = _newton_secant(x0)
    
    if converged and fx_newton is not None and math.isfinite(fx_newton) and abs(fx_newton) <= 1e-8:
        return x_newton
    
    # If Newton didn't converge well, try to bracket
    bracket = _try_bracket(x0)
    
    if bracket is not None:
        a, b, fa, fb = bracket
        if a != b:  # Actual bracket
            x_brent, fx_brent, converged_brent = _brent(a, b, fa, fb)
            if converged_brent and fx_brent is not None and math.isfinite(fx_brent) and abs(fx_brent) <= 1e-8:
                return x_brent
    
    # Return the best candidate we found
    if best_fx is not None and math.isfinite(best_fx):
        return best_x
    
    # Last resort: return x0
    return x0
