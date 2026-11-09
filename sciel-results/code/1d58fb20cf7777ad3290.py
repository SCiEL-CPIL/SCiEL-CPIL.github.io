import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    max_calls = 480
    call_count = 0
    best_x = float(x0)
    best_f = float('inf')

    def time_ok():
        return (time.time() - start_time) < deadline and call_count < max_calls

    def f_call(x):
        nonlocal call_count, best_x, best_f
        call_count += 1
        try:
            fx = f(x)
        except Exception:
            fx = float('nan')
        if math.isfinite(fx) and abs(fx) < abs(best_f):
            best_f = fx
            best_x = x
        return fx

    def secant_method(x1, f1, max_iter=50):
        """Secant method: 1 call per iteration, superlinear convergence."""
        nonlocal best_x, best_f
        x0_s = x1
        f0_s = f1
        # Need a second point; use a small perturbation
        h = 1e-6 * max(1.0, abs(x0_s))
        x1_s = x0_s + h
        f1_s = f_call(x1_s)
        if not math.isfinite(f1_s):
            return None
        
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(f1_s) <= 1e-8:
                return x1_s
            if abs(f0_s) <= 1e-8:
                return x0_s
            
            df = f1_s - f0_s
            if abs(df) < 1e-30:
                break
            
            x_new = x1_s - f1_s * (x1_s - x0_s) / df
            if not math.isfinite(x_new):
                break
            
            # If step is too large, dampen
            if abs(x_new - x1_s) > 100.0 * max(1.0, abs(x1_s)):
                x_new = x1_s + 10.0 * math.copysign(1.0, x_new - x1_s)
            
            f_new = f_call(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Shift: x0 <- x1, x1 <- new
            x0_s, f0_s = x1_s, f1_s
            x1_s, f1_s = x_new, f_new
            
        # Return best of the last few points
        if math.isfinite(f0_s) and abs(f0_s) < abs(best_f):
            return x0_s
        if math.isfinite(f1_s) and abs(f1_s) < abs(best_f):
            return x1_s
        return None

    def brentq(a, fa, b, fb, max_iter=60):
        """Brent's method: hybrid of bisection, secant, and inverse quadratic interpolation."""
        nonlocal best_x, best_f
        # Ensure |fa| <= |fb|
        if abs(fa) > abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        c, fc = a, fa
        d = b - a
        e = d
        
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(fa) <= 1e-8:
                return a
            if abs(fb) <= 1e-8:
                return b
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                return a if abs(fa) < abs(fb) else b
            
            # Check if we can use inverse quadratic interpolation
            use_iqi = False
            x_iqi = None
            if abs(e) > 1e-14 * max(1.0, abs(a), abs(b)):
                # Try inverse quadratic interpolation
                if fa != fb:
                    num1 = fa * (b - a)
                    num2 = fb * (b - a)
                    den = fb - fa
                    # Inverse quadratic interpolation formula
                    x_try = a - (num1 / den) * (fa / (fb - fa) - (b - a) / (b - a))
                    # Use standard IQI formula
                    x_iqi = b - (fb / (fb - fa)) * (b - a)
                    if (2.0 * x_iqi - a - b) * (x_iqi - b) < 0 and abs(x_iqi - b) < 0.75 * abs(b - c):
                        use_iqi = True
            
            if use_iqi and x_iqi is not None:
                x_new = x_iqi
            elif abs(e) > 1e-14 * max(1.0, abs(a), abs(b)) and abs(d) > 1e-14 * max(1.0, abs(a), abs(b)):
                # Secant step
                x_new = b - fb * (b - a) / (fb - fa)
                if (2.0 * x_new - a - b) * (x_new - b) < 0 and abs(x_new - b) < 0.5 * abs(b - c):
                    e = d
                else:
                    x_new = 0.5 * (a + b)
                    e = d
            else:
                # Bisection
                x_new = 0.5 * (a + b)
                e = d
            
            if not math.isfinite(x_new):
                x_new = 0.5 * (a + b)
                e = d
            
            d = b - a
            fb = f_call(x_new)
            if not math.isfinite(fb):
                # Fall back to bisection
                x_new = 0.5 * (a + b)
                fb = f_call(x_new)
                if not math.isfinite(fb):
                    break
            
            if abs(fb) <= 1e-8:
                return x_new
            
            if fa * fb < 0:
                c, fc = b, fb
                b, fb = x_new, fb
            else:
                c, fc = b, fb
                if abs(fa) <= abs(fb):
                    a, fa = x_new, fb
                else:
                    c, fc = x_new, fb
            
        return a if abs(fa) < abs(fb) else b

    def polish(x_start, f_start, max_iter=20):
        """High-precision polish pass using secant with very tight tolerance."""
        nonlocal best_x, best_f
        if not math.isfinite(x_start) or not math.isfinite(f_start):
            return None
        if abs(f_start) <= 1e-14:
            return x_start
        
        x0_s = x_start
        f0_s = f_start
        h = 1e-10 * max(1.0, abs(x0_s))
        x1_s = x0_s + h
        f1_s = f_call(x1_s)
        if not math.isfinite(f1_s):
            return x_start if abs(f_start) <= 1e-8 else None
        
        for _ in range(max_iter):
            if not time_ok():
                break
            if abs(f1_s) <= 1e-14:
                return x1_s
            if abs(f0_s) <= 1e-14:
                return x0_s
            
            df = f1_s - f0_s
            if abs(df) < 1e-30:
                break
            
            x_new = x1_s - f1_s * (x1_s - x0_s) / df
            if not math.isfinite(x_new):
                break
            
            f_new = f_call(x_new)
            if not math.isfinite(f_new):
                break
            if abs(f_new) <= 1e-14:
                return x_new
            
            x0_s, f0_s = x1_s, f1_s
            x1_s, f1_s = x_new, f_new
        
        # Return the best point found
        candidates = [(x0_s, f0_s), (x1_s, f1_s)]
        best_local = None
        best_local_f = float('inf')
        for cx, cf in candidates:
            if math.isfinite(cx) and math.isfinite(cf) and abs(cf) < best_local_f:
                best_local = cx
                best_local_f = cf
        if best_local is not None and abs(best_local_f) <= 1e-8:
            return best_local
        return None

    x = float(x0)
    if not math.isfinite(x):
        return best_x

    fx = f_call(x)
    if not math.isfinite(fx):
        return best_x
    if abs(fx) <= 1e-8:
        return x

    # Strategy 1: Try secant method from x0 with multiple perturbation sizes
    best_secant_result = None
    for h_factor in [1e-6, 1e-4, 1e-2, 1.0]:
        if not time_ok():
            break
        h = h_factor * max(1.0, abs(x))
        x_try = x + h
        fx_try = f_call(x_try)
        if not math.isfinite(fx_try):
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        result = secant_method(x_try, fx_try, max_iter=30)
        if result is not None and math.isfinite(result):
            fr = f(result)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                # Polish for higher precision
                polished = polish(result, fr)
                if polished is not None and math.isfinite(polished):
                    return polished
                return result
        # Also try the other direction
        x_try2 = x - h
        fx_try2 = f_call(x_try2)
        if not math.isfinite(fx_try2):
            continue
        if abs(fx_try2) <= 1e-8:
            return x_try2
        result2 = secant_method(x_try2, fx_try2, max_iter=30)
        if result2 is not None and math.isfinite(result2):
            fr2 = f(result2)
            if math.isfinite(fr2) and abs(fr2) <= 1e-8:
                polished2 = polish(result2, fr2)
                if polished2 is not None and math.isfinite(polished2):
                    return polished2
                return result2

    # Strategy 2: Find a bracket, then use brentq
    a = x
    fa = fx
    h = 1.0
    b = None
    fb = None
    
    for _ in range(25):
        if not time_ok():
            break
        
        b_right = a + h
        fb_right = f_call(b_right)
        if math.isfinite(fb_right) and fa * fb_right < 0:
            b = b_right
            fb = fb_right
            break
        
        b_left = a - h
        fb_left = f_call(b_left)
        if math.isfinite(fb_left) and fa * fb_left < 0:
            b = b_left
            fb = fb_left
            break
        
        # Move to the point with smaller |f|
        if math.isfinite(fb_right) and abs(fb_right) < abs(fa):
            a = b_right
            fa = fb_right
        elif math.isfinite(fb_left) and abs(fb_left) < abs(fa):
            a = b_left
            fa = fb_left
        
        h *= 2.0

    if b is not None:
        if a > b:
            a, b = b, a
            fa, fb = fb, fa
        result = brentq(a, fa, b, fb)
        if result is not None and math.isfinite(result):
            fr = f(result)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                # Polish for higher precision
                polished = polish(result, fr)
                if polished is not None and math.isfinite(polished):
                    return polished
                return result

    # Strategy 3: Try secant from various perturbed points
    for pert in [0.01, -0.01, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]:
        if not time_ok():
            break
        x_try = x0 + pert * max(1.0, abs(x0))
        if not math.isfinite(x_try):
            continue
        fx_try = f_call(x_try)
        if not math.isfinite(fx_try):
            continue
        if abs(fx_try) <= 1e-8:
            return x_try
        result = secant_method(x_try, fx_try, max_iter=30)
        if result is not None and math.isfinite(result):
            fr = f(result)
            if math.isfinite(fr) and abs(fr) <= 1e-8:
                polished = polish(result, fr)
                if polished is not None and math.isfinite(polished):
                    return polished
                return result

    # Final: return best found
    return best_x
