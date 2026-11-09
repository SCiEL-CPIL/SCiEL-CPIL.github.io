"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _safe_f(f, x):
    try:
        y = f(x)
        if not math.isfinite(y):
            return float('inf')
        return float(y)
    except Exception:
        return float('inf')

def _newton_fd(f, x, max_iter=10):
    """
    Newton's method with finite-difference derivative.
    Assumes x is close to a simple root.
    """
    x = float(x)
    for _ in range(max_iter):
        fx = _safe_f(f, x)
        if abs(fx) <= 1e-12:
            return x
        
        h = 1e-5 * max(1.0, abs(x))
        fxh = _safe_f(f, x + h)
        fxh2 = _safe_f(f, x - h)
        
        if math.isfinite(fxh) and math.isfinite(fxh2):
            d = (fxh - fxh2) / (2 * h)
        else:
            d = (fxh - fx) / h if math.isfinite(fxh) else 0.0
            
        if d == 0 or not math.isfinite(d):
            break
            
        x_new = x - fx / d
        if not math.isfinite(x_new):
            break
        x = x_new
    return x

def _illinois_newton_hybrid(f, a, b, fa, fb, max_iter=40):
    """
    Hybrid root finder.
    1. Uses Illinois method (modified bisection) to robustly narrow the bracket.
    2. Once the bracket is small enough, switches to Newton's method for quadratic convergence.
    """
    # Ensure a < b
    if a > b:
        a, b = b, a
        fa, fb = fb, fa

    # If no sign change, we can't bracket.
    # However, the caller (find_root) ensures a sign change before calling this.
    # But just in case, handle gracefully.
    if fa * fb > 0:
        # Return the point with smaller absolute value
        return a if abs(fa) < abs(fb) else b

    for _ in range(max_iter):
        # Check if we can switch to Newton
        # Condition: bracket is small enough that finite difference derivative is reliable
        # and we are close to the root.
        if abs(b - a) < 1e-6 * max(1.0, abs(a), abs(b)):
            # Try Newton from the midpoint or the point with smaller |f|
            x_mid = (a + b) / 2.0
            x_newton = _newton_fd(f, x_mid, max_iter=10)
            fx_newton = _safe_f(f, x_newton)
            fx_mid = _safe_f(f, x_mid)
            
            # If Newton gave a better result and it's finite, use it
            if math.isfinite(fx_newton) and abs(fx_newton) < abs(fx_mid):
                return x_newton
            
            # If Newton didn't help, continue with Illinois
            # Or if we are very close, just return best
            if abs(fx_mid) <= 1e-12:
                return x_mid

        # Illinois Method Step
        # m = (a + b) / 2
        # If f(a) and f(b) have same sign, replace the endpoint with larger |f|
        # with m.
        # Illinois specifically:
        # If |f(a)| < |f(b)|:
        #   b = m
        #   fb = fm
        #   a = a + (b - a) * (fa / (fa - fm))  <-- This is the modified bisection step
        # Else:
        #   a = m
        #   fa = fm
        #   b = b + (a - b) * (fb / (fb - fm))

        m = (a + b) / 2.0
        fm = _safe_f(f, m)
        
        if not math.isfinite(fm):
            # If fm is inf, just bisect
            m = (a + b) / 2.0
            fm = _safe_f(f, m)
            if not math.isfinite(fm):
                break

        if abs(fm) <= 1e-12:
            return m

        if fa * fm <= 0:
            # Root is in [a, m]
            b = m
            fb = fm
            if abs(fa) < abs(fb):
                # fa is smaller, so a is closer to root? No, Illinois adjusts the *other* endpoint.
                # Wait, Illinois adjusts the endpoint with the *larger* absolute value.
                # If |fa| < |fb|, then fb is larger. So we adjust b?
                # Actually, standard Illinois:
                # If f(a) and f(m) have same sign:
                #   a = m
                #   fa = fm
                #   b = b + (a - b) * fb / (fb - fm)
                # If f(b) and f(m) have same sign:
                #   b = m
                #   fb = fm
                #   a = a + (b - a) * fa / (fa - fm)
                
                # Here fa * fm <= 0 means sign change between a and m.
                # So b and m have same sign (or m is 0).
                # So we update b to m? No, the root is in [a, m].
                # So b becomes m.
                # And we adjust a? No, we adjust the endpoint that was *not* updated?
                # Let's re-verify Illinois algorithm.
                
                # Illinois:
                # 1. m = (a+b)/2
                # 2. If f(a)f(m) <= 0: b = m, fb = fm
                #    Else: a = m, fa = fm
                # 3. If |fa| < |fb|:
                #      b = b + (a - b) * fb / (fb - fa)  <-- Wait, this formula uses fa and fb.
                #      Actually: b = b + (a - b) * fb / (fb - fm) is wrong because fm is the new fb or fa.
                
                # Let's use the standard definition:
                # If f(a) and f(m) have the same sign:
                #   a = m
                #   fa = fm
                #   b = b + (a - b) * fb / (fb - fa)  # Here fa is the OLD fa? No, new fa.
                #   Wait, the formula is: b = b + (a - b) * fb / (fb - fm) where fm is the value at m.
                #   But if a was updated, fa is now fm.
                
                # Let's stick to a simpler, robust variant:
                # Modified Bisection (Illinois):
                # If f(a) and f(m) have same sign:
                #   a = m
                #   fa = fm
                #   b = b + (a - b) * fb / (fb - fa)  # fa is now fm
                # Else:
                #   b = m
                #   fb = fm
                #   a = a + (b - a) * fa / (fa - fb)  # fb is now fm
                
                # Let's re-evaluate based on current state:
                # We know fa * fm <= 0. So root in [a, m].
                # So b = m, fb = fm.
                # Now |fa| vs |fb| (which is |fm|).
                # If |fa| < |fb|:
                #   The endpoint b (which is m) has larger value.
                #   We adjust b? No, Illinois adjusts the endpoint with the LARGER value to be closer to the root.
                #   Wait, if |fa| < |fb|, then a is "closer" in value?
                #   Actually, the line passes through (a, fa) and (b, fb).
                #   The x-intercept is x_int = a - fa * (a - b) / (fa - fb).
                #   Illinois sets the new endpoint to x_int.
                #   Which endpoint? The one with the LARGER absolute value.
                #   Here |fb| > |fa|. So we replace b with x_int.
                #   b = a - fa * (a - b) / (fa - fb)
                #   Note: b is currently m.
                pass # Logic below handles it
            else:
                # |fb| <= |fa|
                # Replace a with x_int
                # a = b - fb * (b - a) / (fb - fa)
                pass
        else:
            # fa * fm > 0, so root in [m, b]
            a = m
            fa = fm
            if abs(fa) < abs(fb):
                # |fa| < |fb|, replace b
                # b = a - fa * (a - b) / (fa - fb)
                pass
            else:
                # |fb| <= |fa|, replace a
                # a = b - fb * (b - a) / (fb - fa)
                pass

        # Simplified Illinois implementation to avoid confusion:
        if fa * fm <= 0:
            b = m
            fb = fm
            # Now root is in [a, b]
            if abs(fa) < abs(fb):
                # fb is larger, adjust b
                denom = fb - fa
                if denom != 0:
                    b = a - fa * (a - b) / denom
            else:
                # fa is larger, adjust a
                denom = fb - fa
                if denom != 0:
                    a = b - fb * (b - a) / denom
        else:
            a = m
            fa = fm
            # Now root is in [a, b]
            if abs(fa) < abs(fb):
                # fb is larger, adjust b
                denom = fb - fa
                if denom != 0:
                    b = a - fa * (a - b) / denom
            else:
                # fa is larger, adjust a
                denom = fb - fa
                if denom != 0:
                    a = b - fb * (b - a) / denom

        if not math.isfinite(a) or not math.isfinite(b):
            break
        if abs(b - a) <= 1e-15:
            break

    return (a + b) / 2.0

def find_root(f, x0):
    """
    Global adaptive bracketing search followed by Illinois-Newton refinement.
    """
    start_time = time.time()
    deadline = start_time + 145.0

    x0 = float(x0)
    if not math.isfinite(x0):
        x0 = 0.0

    # Phase 1: Find a bracket [a, b] such that f(a)*f(b) < 0
    # Start with x0 and expand symmetrically.
    a = x0
    fa = _safe_f(f, a)
    
    if not math.isfinite(fa):
        # Try a few random-ish points if x0 is bad
        for dx in [1.0, -1.0, 0.1, -0.1, 10.0, -10.0]:
            a = x0 + dx
            fa = _safe_f(f, a)
            if math.isfinite(fa):
                break
        if not math.isfinite(fa):
            return x0 # Best effort

    # If fa is very small, we might already have a root
    if abs(fa) <= 1e-12:
        return a

    # Expand bracket
    scale = 1.0
    b = None
    fb = None
    
    while time.time() < deadline:
        # Try expanding in both directions
        for direction in [1.0, -1.0]:
            b = x0 + direction * scale
            fb = _safe_f(f, b)
            
            if not math.isfinite(fb):
                continue

            if fa * fb < 0:
                # Found bracket
                if time.time() > deadline:
                    break
                x_root = _illinois_newton_hybrid(f, a, b, fa, fb, max_iter=40)
                fx_root = _safe_f(f, x_root)
                
                # If we found a good root, return it
                if math.isfinite(fx_root) and abs(fx_root) <= 1e-8:
                    return x_root
                
                # If Illinois didn't converge well enough, try Newton from midpoint
                x_mid = (a + b) / 2.0
                x_newton = _newton_fd(f, x_mid, max_iter=10)
                fx_newton = _safe_f(f, x_newton)
                if math.isfinite(fx_newton) and abs(fx_newton) <= 1e-8:
                    return x_newton
                
                # If we still don't have |f(x)| <= 1e-8, return the best we have
                # But the contract says |f(x)| <= 1e-8 is required.
                # So we must keep searching or return the best candidate.
                # However, if we have a bracket, Illinois should get us close.
                # Let's check if we are close enough.
                if abs(fx_root) < abs(fa) and abs(fx_root) < abs(fb):
                     return x_root
                elif abs(fx_newton) < abs(fa) and abs(fx_newton) < abs(fb):
                     return x_newton
                else:
                     # Return the point with smallest |f|
                     return x_root if abs(fx_root) < abs(fx_newton) else x_newton

        # If no bracket found, expand scale
        scale *= 10.0
        if scale > 1e12:
            break

    # If no bracket found, try Newton from x0
    x_newton = _newton_fd(f, x0, max_iter=20)
    fx_newton = _safe_f(f, x_newton)
    
    # Also try a few nearby points
    best_x = x0
    best_fx = fa
    
    if math.isfinite(fx_newton) and abs(fx_newton) < abs(best_fx):
        best_x = x_newton
        best_fx = fx_newton

    for dx in [1e-4, -1e-4, 1e-2, -1e-2, 1.0, -1.0]:
        if time.time() > deadline:
            break
        x_try = x0 + dx
        fx_try = _safe_f(f, x_try)
        if math.isfinite(fx_try) and abs(fx_try) < abs(best_fx):
            best_x = x_try
            best_fx = fx_try
            if abs(fx_try) <= 1e-12:
                break

    return best_x

# EVOLVE-BLOCK-END
