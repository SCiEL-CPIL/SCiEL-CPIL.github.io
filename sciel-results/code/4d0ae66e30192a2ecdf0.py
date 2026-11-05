"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 145.0
    
    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = float('inf')
    
    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() > DEADLINE:
            return None
        evals += 1
        try:
            val = f(x)
            val = float(val)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError):
            return None
        if not math.isfinite(val):
            return None
        abs_val = abs(val)
        if abs_val < best_abs:
            best_abs = abs_val
            best_x = x
            best_f = val
        return val
    
    def is_good(x, fx):
        if fx is None:
            return False
        return abs(fx) <= TOL

    def damped_newton(x0, max_iter=50):
        x = float(x0)
        fx = call_f(x)
        if fx is None:
            return x, None
        if is_good(x, fx):
            return x, fx
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            # Adaptive step for derivative
            h = 1e-8 * max(1.0, abs(x))
            # Try central difference if possible, else forward
            fx_plus = call_f(x + h)
            if fx_plus is None:
                break
            # Check if we can afford central difference
            if evals < MAX_EVALS - 1 and time.monotonic() < DEADLINE:
                fx_minus = call_f(x - h)
                if fx_minus is not None:
                    d = (fx_plus - fx_minus) / (2 * h)
                else:
                    d = (fx_plus - fx) / h
            else:
                d = (fx_plus - fx) / h
            
            if not math.isfinite(d) or abs(d) < 1e-15:
                break
            
            step = fx / d
            x_new = x - step
            
            # Damping: ensure step is not too large
            max_step = 10.0 * max(1.0, abs(x))
            if abs(step) > max_step:
                step = math.copysign(max_step, step)
                x_new = x - step
            
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            
            if is_good(x_new, fx_new):
                return x_new, fx_new
            
            # Check if we made progress
            if abs(fx_new) < abs(fx):
                x = x_new
                fx = fx_new
            else:
                # Halve the step
                step = 0.5 * step
                x_new = x - step
                fx_new = call_f(x_new)
                if fx_new is None:
                    break
                if is_good(x_new, fx_new):
                    return x_new, fx_new
                if abs(fx_new) < abs(fx):
                    x = x_new
                    fx = fx_new
                else:
                    # If still no progress, break
                    break
        
        return x, fx

    def expand_bracket(x0, max_expand=100):
        """
        Expands interval [x0-h, x0+h] until sign change or max evals/time.
        Returns (a, b, fa, fb) if bracket found, else (None, None, None, None)
        """
        h = 1e-4 * max(1.0, abs(x0))
        a = x0 - h
        b = x0 + h
        
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None or fb is None:
            return None, None, None, None
        
        if is_good(a, fa):
            return a, b, fa, fb
        if is_good(b, fb):
            return a, b, fa, fb
        
        # If sign change, we are done
        if fa * fb < 0:
            return a, b, fa, fb
        
        # Expand
        for _ in range(max_expand):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            # Double the step
            h *= 2.0
            a = x0 - h
            b = x0 + h
            
            # Re-evaluate endpoints
            fa = call_f(a)
            if fa is None:
                break
            if is_good(a, fa):
                return a, b, fa, fb
            
            fb = call_f(b)
            if fb is None:
                break
            if is_good(b, fb):
                return a, b, fa, fb
            
            if fa * fb < 0:
                return a, b, fa, fb
            
            # If values are getting huge, maybe we missed it
            if abs(fa) > 1e100 or abs(fb) > 1e100:
                break
        
        return None, None, None, None

    def brent_method(a, b, fa, fb, max_iter=50):
        """
        Assumes fa * fb < 0.
        """
        # Ensure |fa| < |fb| initially for standard Brent logic? 
        # Standard Brent keeps c as the point with smaller |f|
        
        c = a
        fc = fa
        d = b - a
        e = d
        
        # Swap if |fb| > |fa|? No, standard algorithm handles it by swapping a,b,c.
        # Let's stick to a robust version.
        
        for _ in range(max_iter):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            
            if is_good(b, fb):
                return b, fb
            
            # Invariant: fa * fb < 0
            if fa * fb > 0:
                a = c
                fa = fc
                d = b - a
                e = d
            
            if abs(fc) < abs(fb):
                a = b
                b = c
                c = a
                fa = fb
                fb = fc
                fc = fa
            
            tol1 = 2 * 2.22e-16 * abs(b)
            delta = 0.5 * (c - a)
            
            if abs(delta) <= tol1 or fb == 0:
                return b, fb
            
            if abs(e) < tol1 or abs(fa) <= abs(fb):
                # Bisection
                e = delta
                d = fb / fa
            else:
                # Inverse quadratic interpolation
                try:
                    S = fa / fc
                    T1 = fb / fc
                    T2 = fb / fa
                    d = T1 * (T1 - S) * (b - c) / (fa - fc)
                    S = T1 * T2 * (c - a) / (fa - fc)
                    # Check if d is in range
                    if ((d > 0.5 * delta * (c - a)) or (d < 0.5 * delta * (b - c))):
                        d = delta
                    if ((c - b) if (d >= 0) else (b - c)) < (3 * tol1 / 2):
                        # Use bisection if step too small
                        d = delta
                except (ZeroDivisionError, OverflowError):
                    d = delta
            
            e = d
            b_new = b + d
            fb_new = call_f(b_new)
            if fb_new is None:
                break
            
            a = b
            fa = fb
            b = b_new
            fb = fb_new
            
            if fa * fb > 0:
                c = a
                fc = fa
                d = b - a
                e = d
            else:
                c = b
                fc = fb
        
        return b, fb

    # Strategy 1: Damped Newton
    x_res, f_res = damped_newton(x0)
    if is_good(x_res, f_res):
        return x_res
    
    # Strategy 2: Find bracket and use Brent
    a, b, fa, fb = expand_bracket(x0)
    if a is not None and b is not None and fa is not None and fb is not None:
        if is_good(a, fa):
            return a
        if is_good(b, fb):
            return b
        if fa * fb < 0:
            x_res, f_res = brent_method(a, b, fa, fb)
            if is_good(x_res, f_res):
                return x_res
            # If Brent failed to converge to TOL, check if it got close
            if f_res is not None and abs(f_res) < 1e-6:
                 # Try one more Newton step from this point
                 x_res2, f_res2 = damped_newton(x_res, max_iter=10)
                 if is_good(x_res2, f_res2):
                     return x_res2
    
    # Strategy 3: Secant with local points if bracketing failed
    h = 1e-4 * max(1.0, abs(x0))
    x1 = x0 - h
    x2 = x0 + h
    f1 = call_f(x1)
    f2 = call_f(x2)
    
    if f1 is not None and f2 is not None:
        if is_good(x1, f1): return x1
        if is_good(x2, f2): return x2
        
        # Secant iteration
        x_prev, x = x1, x2
        fx_prev, fx = f1, f2
        
        for _ in range(50):
            if time.monotonic() > DEADLINE or evals >= MAX_EVALS - 2:
                break
            if fx == fx_prev:
                break
            x_new = x - fx * (x - x_prev) / (fx - fx_prev)
            if not math.isfinite(x_new):
                break
            fx_new = call_f(x_new)
            if fx_new is None:
                break
            if is_good(x_new, fx_new):
                return x_new
            x_prev, x = x, x_new
            fx_prev, fx = fx, fx_new

    # If all fail, return best x found
    return best_x

# EVOLVE-BLOCK-END
