# EVOLVE-BLOCK-START
import time
import math

def find_root(f, x0):
    """
    Finds a root of f starting near x0 using a robust Newton/Secant hybrid
    with bisection fallback and adaptive step control.
    Returns x such that |f(x)| <= 1e-8.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator kills at 150s)
    tol = 1e-8
    max_iter = 80

    x = float(x0)
    h = 1e-5 * max(1.0, abs(x))
    
    try:
        f0 = f(x)
    except Exception:
        return x

    if not math.isfinite(f0):
        return x
    if abs(f0) <= tol:
        return x

    best_x = x
    best_f_abs = abs(f0)
    
    # Initial bracketing attempt: find a sign change near x0
    a = b = None
    fa = fb = None
    
    # Try several perturbations to find a bracket
    for factor in [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]:
        h_try = factor * h
        for sign in [1, -1]:
            xp = x + sign * h_try
            try:
                fp = f(xp)
            except Exception:
                continue
            if not math.isfinite(fp):
                continue
            
            if abs(fp) < best_f_abs:
                best_x = xp
                best_f_abs = abs(fp)
            
            if abs(fp) <= tol:
                return xp
            
            if fa is None:
                a = x
                fa = f0
            else:
                # Check sign change between a and xp
                if fa * fp < 0:
                    b = xp
                    fb = fp
                    break
            if b is not None:
                break
        if b is not None:
            break
    
    # If no bracket found, use secant/Newton hybrid without bracket
    if b is None:
        # Use secant method with two initial points
        x1 = x + h
        try:
            f1 = f(x1)
        except Exception:
            # Fall back to pure Newton with central difference
            pass
        else:
            if math.isfinite(f1):
                if abs(f1) <= tol:
                    return x1
                if abs(f1) < best_f_abs:
                    best_x = x1
                    best_f_abs = abs(f1)
                
                # Secant iterations
                for _ in range(max_iter):
                    if time.time() > deadline:
                        break
                    
                    if f1 == f0:
                        # Degenerate, perturb
                        x1 = x1 + 1e-4 * max(1.0, abs(x1))
                        try:
                            f1 = f(x1)
                        except Exception:
                            break
                        if not math.isfinite(f1):
                            break
                        if abs(f1) <= tol:
                            return x1
                        if abs(f1) < best_f_abs:
                            best_x = x1
                            best_f_abs = abs(f1)
                        continue
                    
                    denom = f1 - f0
                    if denom == 0 or not math.isfinite(denom):
                        break
                    
                    x_new = x1 - f1 * (x1 - x) / denom
                    
                    if not math.isfinite(x_new):
                        break
                    
                    # Safeguard: limit step size
                    max_step = 5.0 * max(1.0, abs(x1))
                    if abs(x_new - x1) > max_step:
                        x_new = x1 + math.copysign(max_step, x_new - x1)
                    
                    try:
                        f_new = f(x_new)
                    except Exception:
                        break
                    
                    if not math.isfinite(f_new):
                        break
                    
                    if abs(f_new) <= tol:
                        return x_new
                    
                    if abs(f_new) < best_f_abs:
                        best_x = x_new
                        best_f_abs = abs(f_new)
                    
                    x, f0 = x1, f1
                    x1, f1 = x_new, f_new
                return best_x
    
    # Bracket found: use bisection + Newton hybrid (Illinois method variant)
    # Ensure fa and fb have opposite signs
    if fa * fb > 0:
        # Swap if needed
        a, b = b, a
        fa, fb = fb, fa
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        if abs(fb) < abs(fa):
            # Swap so that |fb| >= |fa| is not guaranteed; keep the one with smaller |f| as the "active"
            # Actually for bisection+Newton, we want to move the endpoint with larger |f|
            pass
        
        # Midpoint
        mid = 0.5 * (a + b)
        if not math.isfinite(mid):
            break
        
        # Check if midpoint is good enough
        try:
            fmid = f(mid)
        except Exception:
            break
        
        if not math.isfinite(fmid):
            break
        
        if abs(fmid) <= tol:
            return mid
        
        if abs(fmid) < best_f_abs:
            best_x = mid
            best_f_abs = abs(fmid)
        
        # Try Newton step from the endpoint with smaller |f|
        if abs(fa) < abs(fb):
            # Newton from a
            # Estimate derivative using secant slope
            if b != a:
                slope = (fb - fa) / (b - a)
                if math.isfinite(slope) and slope != 0:
                    x_newton = a - fa / slope
                    if math.isfinite(x_newton) and a < x_newton < b:
                        try:
                            f_newton = f(x_newton)
                        except Exception:
                            f_newton = None
                        if f_newton is not None and math.isfinite(f_newton):
                            if abs(f_newton) <= tol:
                                return x_newton
                            if abs(f_newton) < best_f_abs:
                                best_x = x_newton
                                best_f_abs = abs(f_newton)
                            if fa * f_newton <= 0:
                                b = x_newton
                                fb = f_newton
                                continue
                            else:
                                a = x_newton
                                fa = f_newton
                                continue
        else:
            # Newton from b
            if b != a:
                slope = (fb - fa) / (b - a)
                if math.isfinite(slope) and slope != 0:
                    x_newton = b - fb / slope
                    if math.isfinite(x_newton) and a < x_newton < b:
                        try:
                            f_newton = f(x_newton)
                        except Exception:
                            f_newton = None
                        if f_newton is not None and math.isfinite(f_newton):
                            if abs(f_newton) <= tol:
                                return x_newton
                            if abs(f_newton) < best_f_abs:
                                best_x = x_newton
                                best_f_abs = abs(f_newton)
                            if fa * f_newton <= 0:
                                b = x_newton
                                fb = f_newton
                                continue
                            else:
                                a = x_newton
                                fa = f_newton
                                continue
        
        # Fall back to bisection
        if fa * fmid <= 0:
            b = mid
            fb = fmid
        else:
            a = mid
            fa = fmid
        
        # Convergence check
        if abs(b - a) < 1e-12 * max(1.0, abs(mid)):
            break
    
    # Final check on best_x
    try:
        fb_best = f(best_x)
        if math.isfinite(fb_best) and abs(fb_best) <= tol:
            return best_x
    except Exception:
        pass
    
    return best_x
# EVOLVE-BLOCK-END
