import math
import time

def find_root(f, x0):
    """
    Find a root x of f such that |f(x)| <= 1e-8.
    Primary strategy: Inverse Quadratic Interpolation (IQI) with safeguards.
    Fallback: Bisection with expansion.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    x0 = float(x0)
    
    # Helper to check time
    def time_up():
        return time.time() > deadline

    # 1. Initial Evaluation
    try:
        fx0 = f(x0)
    except Exception:
        return x0

    if not math.isfinite(fx0):
        return x0

    if abs(fx0) <= 1e-8:
        return x0

    # 2. Bracketing Search
    # We need two points with opposite signs to use IQI/Bisection robustly.
    # Start with a small step and expand exponentially.
    scale = max(1.0, abs(x0))
    h = 1e-4 * scale
    
    a, b = None, None
    fa, fb = None, None
    
    # We have (x0, fx0). Let's call it point C initially.
    # We want to find a point D such that sign(f(D)) != sign(f(C)).
    # Then we have a bracket [a, b].
    
    # Try left first, then right, expanding.
    found_bracket = False
    
    # We'll keep a list of points to potentially use for IQI if we don't find a bracket immediately
    # but IQI usually requires a bracket or a very good local model.
    # Let's stick to finding a bracket first for robustness, then use IQI to converge fast.
    
    # Strategy:
    # 1. Evaluate x0.
    # 2. Try x0 +/- h, x0 +/- 10h, x0 +/- 100h, etc.
    # 3. Once a bracket [a, b] is found where f(a)*f(b) < 0, use IQI.
    
    # To minimize calls, we evaluate in a specific order.
    # Since we don't know the period, we just expand.
    
    # Current known points: (x0, fx0)
    # Let's try to find a neighbor with opposite sign.
    
    points = [(x0, fx0)]
    
    # Expand left and right
    step = h
    max_expansions = 20 # 1e-4 * 10^20 is huge, should be enough for any reasonable root
    
    while not found_bracket and not time_up():
        # Try left
        x_left = x0 - step
        try:
            f_left = f(x_left)
            if math.isfinite(f_left):
                if abs(f_left) <= 1e-8:
                    return x_left
                if fx0 * f_left < 0:
                    a, b = x_left, x0
                    fa, fb = f_left, fx0
                    found_bracket = True
                    break
                points.append((x_left, f_left))
        except Exception:
            pass
            
        # Try right
        x_right = x0 + step
        try:
            f_right = f(x_right)
            if math.isfinite(f_right):
                if abs(f_right) <= 1e-8:
                    return x_right
                if fx0 * f_right < 0:
                    a, b = x0, x_right
                    fa, fb = fx0, f_right
                    found_bracket = True
                    break
                points.append((x_right, f_right))
        except Exception:
            pass
            
        step *= 10.0
        if step > 1e12:
            break

    if not found_bracket:
        # If we couldn't find a bracket, maybe the function doesn't change sign?
        # Or we ran out of time/space.
        # Try a local minimization approach? Or just return the point with smallest |f|.
        best_x = x0
        best_f = abs(fx0)
        for px, pf in points:
            if abs(pf) < best_f:
                best_f = abs(pf)
                best_x = px
        return best_x

    # 3. IQI / Bisection Hybrid
    # We have a bracket [a, b] with fa, fb.
    # We also have a third point? 
    # IQI needs 3 points. We have a and b. 
    # We can use the midpoint or one of the previous points as the third.
    # Let's initialize the third point as the midpoint to start bisection, 
    # but IQI converges faster if we have a good third point.
    # Actually, standard IQI uses the last 3 points. 
    # Let's start with a, b, and the midpoint m.
    
    m = (a + b) / 2.0
    try:
        fm = f(m)
    except Exception:
        fm = None
    
    if fm is not None and math.isfinite(fm):
        if abs(fm) <= 1e-8:
            return m
        # Update bracket if necessary (shouldn't be needed if bisection midpoint)
        # But for IQI, we need 3 points.
        # Points: (a, fa), (b, fb), (m, fm)
        
        # We will iterate.
        # State: a, b, fa, fb. And a "last" point c, fc which is the most recent evaluation.
        # Initially, let c = m, fc = fm.
        
        c = m
        fc = fm
        
        # Ensure a, b, c are distinct enough
        # Standard IQI step:
        # Fit quadratic through (a,fa), (b,fb), (c,fc).
        # Solve for x where q(x)=0.
        
        for _ in range(100): # Max iterations
            if time_up():
                break
                
            # Check if we can do IQI
            # Conditions:
            # 1. a, b, c are distinct
            # 2. The quadratic has a real root in [a, b] (or close)
            # 3. The step is not too small (to avoid division by zero)
            
            if abs(a - b) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
                
            # Lagrange interpolation for inverse function x(y)
            # We want x such that q(x) = 0.
            # q(x) is the quadratic interpolant.
            # It's easier to interpolate x as a function of y (inverse quadratic).
            # x = y0 * L0(y) + y1 * L1(y) + y2 * L2(y) where y is the target (0).
            # Actually, let's just solve the quadratic for x directly?
            # No, inverse quadratic interpolation (IQI) models x = P(y).
            # We have points (fa, a), (fb, b), (fc, c).
            # We want x such that P(x) = 0? No, P(y) = x. We want P(0).
            
            # Let's use the formula for the next point x_next in IQI.
            # x_next = (fa^2 * fb * fc * (b - c) / ((fa - fb) * (fa - fc)) +
            #           fb^2 * fa * fc * (c - a) / ((fb - fa) * (fb - fc)) +
            #           fc^2 * fa * fb * (a - b) / ((fc - fa) * (fc - fb)))
            
            # This formula assumes fa, fb, fc are distinct and non-zero.
            
            if abs(fa - fb) < 1e-15 or abs(fa - fc) < 1e-15 or abs(fb - fc) < 1e-15:
                # Fall back to bisection
                mid = (a + b) / 2.0
                try:
                    fmid = f(mid)
                except Exception:
                    break
                if not math.isfinite(fmid):
                    break
                if abs(fmid) <= 1e-8:
                    return mid
                if fa * fmid < 0:
                    b, fb = mid, fmid
                else:
                    a, fa = mid, fmid
                c, fc = mid, fmid
                continue
            
            # Calculate IQI step
            try:
                denom1 = (fa - fb) * (fa - fc)
                denom2 = (fb - fa) * (fb - fc)
                denom3 = (fc - fa) * (fc - fb)
                
                term1 = (fa * fa * fb * fc * (b - c)) / denom1
                term2 = (fb * fb * fa * fc * (c - a)) / denom2
                term3 = (fc * fc * fa * fb * (a - b)) / denom3
                
                x_iqi = term1 + term2 + term3
            except Exception:
                x_iqi = None
            
            # Safeguards:
            # 1. If x_iqi is not finite, use bisection.
            # 2. If x_iqi is outside [a, b], use bisection.
            # 3. If x_iqi is too close to a or b, use bisection.
            
            use_bisection = False
            
            if x_iqi is None or not math.isfinite(x_iqi):
                use_bisection = True
            else:
                # Check if in bracket
                if x_iqi <= a or x_iqi >= b:
                    use_bisection = True
                else:
                    # Check if step is too small
                    dist_to_a = abs(x_iqi - a)
                    dist_to_b = abs(x_iqi - b)
                    min_dist = min(dist_to_a, dist_to_b)
                    bracket_width = abs(b - a)
                    if min_dist < 1e-14 * bracket_width:
                        use_bisection = True
            
            if use_bisection:
                x_next = (a + b) / 2.0
            else:
                x_next = x_iqi
            
            # Evaluate f(x_next)
            try:
                f_next = f(x_next)
            except Exception:
                break
            
            if not math.isfinite(f_next):
                break
                
            if abs(f_next) <= 1e-8:
                return x_next
            
            # Update bracket
            if fa * f_next < 0:
                b, fb = x_next, f_next
            else:
                a, fa = x_next, f_next
            
            # Update "last" point for IQI
            # We want the 3 points to be a, b, and the one that is NOT the new midpoint?
            # Standard practice: replace the point that is not the new bracket endpoint?
            # Actually, we have a, b (bracket) and c (previous).
            # The new point x_next is now one of the endpoints.
            # We should keep the other endpoint and the previous c?
            # Or better: keep the two most recent points and the other endpoint?
            # Let's just set c = x_next, fc = f_next for simplicity, 
            # but this might degrade if x_next is always bisection.
            # A better strategy: keep the point that is NOT the one just replaced in the bracket?
            # If we updated b, then a is the other endpoint. c was the previous point.
            # Let's keep a, b, and c. If we updated b, the new set is a, b_new, c.
            # If we updated a, the new set is a_new, b, c.
            # We should probably drop the oldest point or the one furthest from the root?
            # For simplicity and robustness, let's just keep the last evaluated point as c.
            c, fc = x_next, f_next
            
            # Convergence check
            if abs(b - a) < 1e-14 * max(1.0, abs(a), abs(b)):
                break
        
        # Return the midpoint of the final bracket or the last point
        return (a + b) / 2.0

    else:
        # If we couldn't evaluate midpoint, just return x0 or best available
        return x0
