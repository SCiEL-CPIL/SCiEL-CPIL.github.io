import math
import time

def find_root(f, x0):
    start_time = time.time()
    deadline = start_time + 110.0
    
    def time_ok():
        return time.time() < deadline

    x = float(x0)
    
    # 1. Initial Evaluation
    try:
        fx = f(x)
    except Exception:
        return x0

    if not math.isfinite(fx):
        # Search for a finite point
        found = False
        for dx in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e5]:
            for sign in [1, -1]:
                if not time_ok():
                    break
                x_test = x + sign * dx
                try:
                    fx_test = f(x_test)
                except Exception:
                    continue
                if math.isfinite(fx_test):
                    x = x_test
                    fx = fx_test
                    found = True
                    break
            if found or not time_ok():
                break
        if not found:
            return x0
        if abs(fx) <= 1e-8:
            return x

    if abs(fx) <= 1e-8:
        return x

    best_x = x
    best_fx = fx
    best_abs_fx = abs(fx)

    # 2. Bracketing Phase
    # Try to find a sign change in a local neighborhood
    a = x
    fa = fx
    b = None
    fb = None
    
    # Expand to the right
    step = max(1e-4, abs(x) * 0.1)
    for i in range(50):
        if not time_ok():
            break
        if step > 1e8:
            break
        b_test = a + step
        try:
            fb_test = f(b_test)
        except Exception:
            step *= 2
            continue
        if math.isfinite(fb_test):
            if fa * fb_test < 0:
                b = b_test
                fb = fb_test
                break
            elif abs(fb_test) < best_abs_fx:
                best_x = b_test
                best_fx = fb_test
                best_abs_fx = abs(fb_test)
                # Move anchor to the better point to keep expanding in that direction?
                # No, keep expanding from original a to ensure we cross the root if it's further out.
                # But if we move a, we lose the sign change context.
                # Standard expansion: keep a fixed, increase step.
            step *= 1.5
        else:
            step *= 2

    if b is None:
        # Expand to the left
        step = max(1e-4, abs(x) * 0.1)
        for i in range(50):
            if not time_ok():
                break
            if step > 1e8:
                break
            a_test = x - step
            try:
                fa_test = f(a_test)
            except Exception:
                step *= 2
                continue
            if math.isfinite(fa_test):
                if fx * fa_test < 0:
                    a = a_test
                    fa = fa_test
                    b = x
                    fb = fx
                    break
                elif abs(fa_test) < best_abs_fx:
                    best_x = a_test
                    best_fx = fa_test
                    best_abs_fx = abs(fa_test)
            step *= 1.5
        else:
            # No bracket found. Assume multiple root or no root in range.
            # Use minimization of |f(x)| via Parabolic Interpolation
            return _minimize_abs_f(f, x, fx, best_x, best_fx, best_abs_fx, deadline)

    # 3. Root Finding with Bracket [a, b]
    # Use IQI (Inverse Quadratic Interpolation) with Bisection fallback
    # We need 3 points. Start with Secant to generate the 3rd, or just use Secant for first few steps.
    # Actually, IQI works well if we have 3 points. Let's initialize with x0, a, b.
    # But x0 might be inside [a,b].
    
    # Let's use a robust hybrid: 
    # Points: x1, x2 (the two most recent)
    # If we have a bracket [lo, hi], we ensure the new point stays in [lo, hi].
    
    lo, hi = a, b
    flo, fhi = fa, fb
    
    # Ensure |flo| <= |fhi| is not strictly necessary for IQI, but helpful for Secant.
    # For IQI, we need 3 distinct points.
    # Let's pick a third point. Midpoint is safe.
    x_mid = 0.5 * (lo + hi)
    try:
        f_mid = f(x_mid)
    except Exception:
        return best_x if best_abs_fx < 1e-8 else (lo if abs(flo) < abs(fhi) else hi)
    
    if not math.isfinite(f_mid):
        return best_x if best_abs_fx < 1e-8 else (lo if abs(flo) < abs(fhi) else hi)

    if abs(f_mid) < best_abs_fx:
        best_x = x_mid
        best_fx = f_mid
        best_abs_fx = abs(f_mid)
    
    if abs(f_mid) <= 1e-8:
        return x_mid

    # Now we have 3 points: (lo, flo), (hi, fhi), (x_mid, f_mid)
    # We will iterate.
    # State: x1, x2, x3 are the last 3 points.
    # To keep it simple and robust, let's maintain a list of points and use the last 3 for IQI.
    # However, IQI can be unstable if points are close.
    # Alternative: Use the "Illinois" algorithm which is a modified Regula Falsi. It's very robust and faster than standard Bisection.
    # But IQI is faster. Let's try IQI with safeguards.
    
    # Initialize points for IQI
    # Sort points by x to maintain order? IQI doesn't require order, but it's good for debugging.
    # Let's just use the 3 points we have.
    x1, f1 = lo, flo
    x2, f2 = hi, fhi
    x3, f3 = x_mid, f_mid
    
    # Ensure we track the bracket [lo, hi] correctly.
    # The bracket is defined by the two points with opposite signs that enclose the root.
    # Initially lo, hi.
    # If we introduce x_mid, we must update the bracket if x_mid has a sign change with lo or hi.
    
    # Let's switch to a cleaner implementation:
    # Keep lo, hi as the current bracket with sign change.
    # Keep x_prev, f_prev as the last point used (outside or inside bracket).
    # Actually, IQI uses the inverse function.
    # x_new = q(f_new) where q is the quadratic interpolant of (f_i, x_i).
    
    # Let's use a simpler, very robust method: **Brent's Method** logic is complex to implement from scratch correctly in short code.
    # Let's stick to **Secant with Bisection Fallback** but optimized.
    # Wait, the prompt says "IQI exploits polynomial nature... significantly reducing calls".
    # So IQI is the way.
    
    # Implementation of IQI:
    # Given (x1, y1), (x2, y2), (x3, y3), find x where y=0.
    # The quadratic in y passing through these points is:
    # x(y) = c0 + c1*y + c2*y^2
    # We can compute the coefficients or use the Lagrange form.
    # x(y) = x1 * L1(y) + x2 * L2(y) + x3 * L3(y)
    # L1(y) = (y-y2)(y-y3) / ((y1-y2)(y1-y3))
    # ...
    # We want x(0).
    # x(0) = x1 * (-y2)(-y3) / ((y1-y2)(y1-y3)) + x2 * (-y1)(-y3) / ((y2-y1)(y2-y3)) + x3 * (-y1)(-y2) / ((y3-y1)(y3-y2))
    # x(0) = x1 * y2*y3 / ((y1-y2)(y1-y3)) + x2 * y1*y3 / ((y2-y1)(y2-y3)) + x3 * y1*y2 / ((y3-y1)(y3-y2))
    
    def iqi_step(x1, y1, x2, y2, x3, y3):
        # Check for division by zero
        if (y1 - y2) == 0 or (y1 - y3) == 0 or (y2 - y3) == 0:
            return None
        term1 = x1 * y2 * y3 / ((y1 - y2) * (y1 - y3))
        term2 = x2 * y1 * y3 / ((y2 - y1) * (y2 - y3))
        term3 = x3 * y1 * y2 / ((y3 - y1) * (y3 - y2))
        return term1 + term2 + term3

    # Loop
    for i in range(100):
        if not time_ok():
            break
        
        # Update best
        if abs(flo) < best_abs_fx:
            best_x = lo
            best_fx = flo
            best_abs_fx = abs(flo)
        if abs(fhi) < best_abs_fx:
            best_x = hi
            best_fx = fhi
            best_abs_fx = abs(fhi)
            
        if best_abs_fx <= 1e-8:
            return best_x

        # Check convergence of bracket
        if abs(hi - lo) < 1e-12 * max(1.0, abs(hi), abs(lo)):
            return lo if abs(flo) < abs(fhi) else hi

        # Try IQI
        # We need 3 points. We have lo, hi, and the last point we tried.
        # Let's maintain a history of the last 3 points used in the iteration.
        # Initially: lo, hi, x_mid.
        # In subsequent steps, we replace one point.
        
        # Let's use the points: lo, hi, and x_new_prev
        # We need to store x_new_prev.
        # Let's restructure to keep 3 points: p1, p2, p3.
        
        # For simplicity and robustness, let's use **Secant** if we don't have 3 good points, 
        # and **IQI** if we do.
        # But managing the 3 points and bracket is tricky.
        
        # Let's use a simpler robust hybrid:
        # 1. Calculate Secant point.
        # 2. Calculate IQI point (if possible).
        # 3. If IQI point is within [lo, hi], use it. Else use Secant (if in [lo, hi]) or Bisection.
        
        # We need a third point for IQI. Let's keep `x_last` and `f_last`.
        # Initially x_last = x_mid, f_last = f_mid.
        
        # We need to define x_last, f_last outside the loop.
        # Let's move initialization here.
        
        pass # Placeholder for logic below

    # Re-implementing the loop properly with state variables
    x_last = x_mid
    f_last = f_mid
    
    for i in range(100):
        if not time_ok():
            break
        
        # Check best
        if abs(flo) < best_abs_fx:
            best_x = lo; best_fx = flo; best_abs_fx = abs(flo)
        if abs(fhi) < best_abs_fx:
            best_x = hi; best_fx = fhi; best_abs_fx = abs(fhi)
            
        if best_abs_fx <= 1e-8:
            return best_x

        if abs(hi - lo) < 1e-12 * max(1.0, abs(hi), abs(lo)):
            return lo if abs(flo) < abs(fhi) else hi

        # 1. Secant Point
        if abs(fhi - flo) > 1e-14:
            x_sec = lo - flo * (hi - lo) / (fhi - flo)
        else:
            x_sec = 0.5 * (lo + hi)

        # 2. IQI Point
        x_iqi = None
        # Check if we have 3 distinct y values
        if abs(flo - fhi) > 1e-14 and abs(flo - f_last) > 1e-14 and abs(fhi - f_last) > 1e-14:
            try:
                x_iqi = iqi_step(lo, flo, hi, fhi, x_last, f_last)
            except Exception:
                x_iqi = None

        # 3. Choose Point
        x_new = None
        # Prefer IQI if it's inside the bracket
        if x_iqi is not None and math.isfinite(x_iqi) and lo < x_iqi < hi:
            x_new = x_iqi
        else:
            # Prefer Secant if inside bracket
            if math.isfinite(x_sec) and lo < x_sec < hi:
                x_new = x_sec
            else:
                # Fallback to Bisection
                x_new = 0.5 * (lo + hi)

        try:
            f_new = f(x_new)
        except Exception:
            break

        if not math.isfinite(f_new):
            break

        if abs(f_new) < best_abs_fx:
            best_x = x_new
            best_fx = f_new
            best_abs_fx = abs(f_new)

        if abs(f_new) <= 1e-8:
            return x_new

        # Update Bracket
        if flo * f_new < 0:
            hi, fhi = x_new, f_new
        else:
            lo, flo = x_new, f_new

        # Update History for IQI
        # We want to keep the 3 points that are most "informative".
        # Usually, the two bracket endpoints and the last point.
        # So, x_last should be the point we just replaced?
        # Actually, for IQI, using the two bracket endpoints and the *previous* point is standard.
        # So, x_last, f_last should become the point that was dropped from the bracket?
        # No, just keep the last evaluated point as the 3rd point for the next iteration.
        x_last, f_last = x_new, f_new

    return best_x

def _minimize_abs_f(f, x0, fx0, best_x, best_fx, best_abs_fx, deadline):
    # Minimize |f(x)| using Parabolic Interpolation
    # Start with x0, and two neighbors
    step = max(1e-4, abs(x0) * 0.1)
    x1 = x0 - step
    x2 = x0
    x3 = x0 + step
    
    try:
        f1 = abs(f(x1))
        f2 = abs(f(x2))
        f3 = abs(f(x3))
    except Exception:
        return best_x

    if not (math.isfinite(f1) and math.isfinite(f2) and math.isfinite(f3)):
        return best_x

    for i in range(50):
        if not time_ok(deadline):
            break
            
        # Find min of f1, f2, f3
        min_val = min(f1, f2, f3)
        if min_val < best_abs_fx:
            if f1 == min_val:
                best_x = x1; best_abs_fx = f1
            elif f2 == min_val:
                best_x = x2; best_abs_fx = f2
            else:
                best_x = x3; best_abs_fx = f3
                
        if best_abs_fx <= 1e-8:
            return best_x
            
        # Parabolic interpolation
        # Find vertex of parabola through (x1,f1), (x2,f2), (x3,f3)
        # x_new = x2 + 0.5 * ( (x2-x1)^2(f2-f3) - (x2-x3)^2(f2-f1) ) / ( (x2-x1)(f2-f3) - (x2-x3)(f2-f1) )
        denom = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if abs(denom) < 1e-14:
            # Fallback to golden section or bisection
            # Just move a bit
            x_new = x2
        else:
            num = 0.5 * ((x2 - x1)**2 * (f2 - f3) - (x2 - x3)**2 * (f2 - f1))
            x_new = x2 + num / denom
            
        # Safeguard: ensure x_new is close
        if not math.isfinite(x_new):
            break
            
        # Evaluate
        try:
            f_new = abs(f(x_new))
        except Exception:
            break
            
        if not math.isfinite(f_new):
            break
            
        # Update points
        if f_new < f2:
            # New point is better
            if x_new < x2:
                x3, f3 = x2, f2
                x2, f2 = x_new, f_new
            else:
                x1, f1 = x2, f2
                x2, f2 = x_new, f_new
        else:
            # New point is worse, discard the worst of the old 3
            if f1 > f3:
                x1, f1 = x_new, f_new
            else:
                x3, f3 = x_new, f_new
                
    return best_x

def time_ok(deadline):
    return time.time() < deadline
