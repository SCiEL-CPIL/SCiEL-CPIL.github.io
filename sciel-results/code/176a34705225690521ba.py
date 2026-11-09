import math
import time

def find_root(f, x0):
    """
    Find root x of f(x) such that |f(x)| <= 1e-8.
    Uses a hybrid approach: secant leading with IQI polish, or bracketed Brent if needed.
    """
    start_time = time.time()
    deadline = start_time + 100.0  # Safety margin well under 120s
    tol = 1e-8  # The actual requirement
    tol_inner = 1e-10  # Slightly tighter for safety
    
    x0 = float(x0)
    f0 = f(x0)
    if abs(f0) <= tol:
        return x0

    # Strategy: Lead with secant to avoid bracketing overhead when x0 is near a root.
    # Use IQI (Inverse Quadratic Interpolation) for superlinear convergence on smooth functions.
    # Fall back to bisection if secant/IQI diverges or if we can find a bracket.
    
    # Initialize secant points
    # Step 1: Choose a second point. Use a small step relative to x0.
    # Adaptive step: if x0 is large, step is larger.
    step = 1e-4 * max(1.0, abs(x0))
    
    x_prev = x0
    f_prev = f0
    x_curr = x0 + step
    f_curr = f(x_curr)
    
    if abs(f_curr) <= tol:
        return x_curr

    # Track best point found
    best_x = x_prev if abs(f_prev) < abs(f_curr) else x_curr
    best_f = min(abs(f_prev), abs(f_curr))

    # History for IQI: (x, f(x))
    # We need 3 distinct points for IQI.
    # Start with 2 points.
    history = [(x_prev, f_prev), (x_curr, f_curr)]
    
    # If the first two points have the same sign, we might need to look for a bracket
    # or just continue secant. Secant can work without a bracket if it converges.
    # But for robustness, if secant fails to converge in a few steps, try bracketing.
    
    max_iter = 100
    secant_fails = 0
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Check if we've found a root
        if abs(f_curr) <= tol:
            return x_curr
        
        if abs(f_prev) < best_f:
            best_x = x_prev
            best_f = abs(f_prev)
        if abs(f_curr) < best_f:
            best_x = x_curr
            best_f = abs(f_curr)

        # Calculate next point
        # Try IQI if we have 3 distinct points
        x_next = None
        if len(history) >= 3:
            (x1, f1), (x2, f2), (x3, f3) = history[-3]
            # Ensure points are distinct
            if x1 != x2 and x2 != x3 and x1 != x3:
                # IQI formula
                # x = (x1*x2*f3 + x2*x3*f1 + x3*x1*f2) / (x2*f3 + x3*f1 + x1*f2 - x1*f2 - x2*f1 - x3*f2) ... wait
                # Correct Lagrange basis for x at y=0:
                # L1 = x1 * (f2*f3) / ((f1-f2)*(f1-f3))
                # L2 = x2 * (f1*f3) / ((f2-f1)*(f2-f3))
                # L3 = x3 * (f1*f2) / ((f3-f1)*(f3-f2))
                # x = L1 + L2 + L3
                try:
                    d1 = (f1 - f2) * (f1 - f3)
                    d2 = (f2 - f1) * (f2 - f3)
                    d3 = (f3 - f1) * (f3 - f2)
                    if abs(d1) > 1e-15 and abs(d2) > 1e-15 and abs(d3) > 1e-15:
                        L1 = x1 * (f2 * f3) / d1
                        L2 = x2 * (f1 * f3) / d2
                        L3 = x3 * (f1 * f2) / d3
                        x_next = L1 + L2 + L3
                except (ZeroDivisionError, OverflowError):
                    pass

        # If IQI failed or we don't have 3 points, use Secant
        if x_next is None or not math.isfinite(x_next):
            denom = f_curr - f_prev
            if abs(denom) < 1e-15:
                # Slope is zero, try a different step or break
                # Perturb the step
                step *= 10
                x_curr = x_prev + step
                f_curr = f(x_curr)
                history.append((x_curr, f_curr))
                secant_fails += 1
                if secant_fails > 5:
                    break
                continue
            x_next = x_curr - f_curr * (x_curr - x_prev) / denom

        # Safeguard: If x_next is far away, it might be diverging.
        # Limit the step size to prevent runaway.
        # If |x_next - x_curr| > 10 * |x_curr - x_prev|, cap it.
        if abs(x_next - x_curr) > 10.0 * abs(x_curr - x_prev):
             # Use a smaller step, maybe bisection-like if we had a bracket, 
             # but here we just shrink the secant step.
             # Actually, if it's diverging, let's just try a smaller perturbation from the best point.
             # Or just accept it and see if it comes back? No, that's risky.
             # Let's cap the movement.
             direction = 1 if x_next > x_curr else -1
             x_next = x_curr + direction * 10.0 * abs(x_curr - x_prev)
             
        f_next = f(x_next)
        
        if not math.isfinite(f_next):
            # Diverged. Try to find a bracket around best_x.
            # This is a fallback.
            a = best_x - 1e-4 * max(1.0, abs(best_x))
            b = best_x + 1e-4 * max(1.0, abs(best_x))
            fa = f(a)
            fb = f(b)
            if math.isfinite(fa) and math.isfinite(fb) and fa * fb < 0:
                # Found a bracket, switch to Brent/Bisection
                return _brent(f, a, fa, b, fb, tol, deadline)
            else:
                # Give up secant, return best
                break

        # Update history
        # We want to keep the 3 most recent points for IQI, but ensure they are distinct.
        # Standard practice: replace the point with the smallest |f|? No, usually replace the oldest.
        # For IQI, we need 3 points. Let's just append and keep last 3.
        history.append((x_next, f_next))
        if len(history) > 3:
            history.pop(0)

        # Update secant points
        # If f_next is smaller than f_prev, we are making progress.
        # x_prev, f_prev = x_curr, f_curr
        # x_curr, f_curr = x_next, f_next
        # This is standard secant update.
        
        # Check for convergence of the interval? We don't have a bracket.
        # Just check if x_next is close to x_curr.
        if abs(x_next - x_curr) < 1e-15:
            break

        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next

    return best_x

def _brent(f, a, fa, b, fb, tol, deadline):
    """
    Brent's method with bisection fallback.
    Assumes f(a)*f(b) < 0.
    """
    # Ensure fa is the smaller magnitude
    if abs(fa) > abs(fb):
        a, b = b, a
        fa, fb = fb, fa
        
    c = a
    fc = fa
    d = b - a
    e = d
    CONVERGENCE = 1e-12
    
    for _ in range(100):
        if time.time() > deadline:
            break
            
        if abs(fb) <= tol:
            return b
        if abs(fc) <= tol:
            return c
            
        # Check for convergence
        if abs(b - a) < tol:
            return (a + b) / 2.0

        # Inverse Quadratic Interpolation
        if a != c and c != b:
            # Lagrange interpolation
            try:
                t1 = (fa - fb) * (fa - fc)
                t2 = (fb - fa) * (fb - fc)
                t3 = (fc - fa) * (fc - fb)
                if abs(t1) > 1e-15 and abs(t2) > 1e-15 and abs(t3) > 1e-15:
                    L1 = a * (fb * fc) / t1
                    L2 = b * (fa * fc) / t2
                    L3 = c * (fa * fb) / t3
                    s = L1 + L2 + L3
                else:
                    s = None
            except (ZeroDivisionError, OverflowError):
                s = None
        else:
            s = None

        # Check if IQI is acceptable
        if s is not None:
            # s should be in (a, b)
            if (s > a and s < b) or (s < a and s > b):
                # Check if s is not too close to the midpoint
                mid = (a + b) / 2.0
                if abs(s - mid) < 0.5 * abs(b - a):
                    # Accept IQI
                    pass
                else:
                    # Use secant
                    s = b - fb * (b - a) / (fb - fa) if fb != fa else mid
            else:
                # Use secant
                s = b - fb * (b - a) / (fb - fa) if fb != fa else mid
        else:
            # Use secant
            s = b - fb * (b - a) / (fb - fa) if fb != fa else (a + b) / 2.0

        if not math.isfinite(s):
            s = (a + b) / 2.0

        # Ensure s is strictly inside the bracket
        if not ((s > a and s < b) or (s < a and s > b)):
            s = (a + b) / 2.0

        fs = f(s)
        
        if not math.isfinite(fs):
            break

        if fs == 0:
            return s

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
            
        if abs(fa) < abs(fb):
            c, fc = b, fb
            b, fb = a, fa
            a, fa = c, fc
            
        d = b - a
        e = b - a

    # Return the best estimate
    if abs(fa) < abs(fb):
        return a
    else:
        return b
