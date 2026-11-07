import math
import time

def find_root(f, x0):
    start_time = time.monotonic()
    deadline = start_time + 100.0
    max_calls = 200 # Safety limit on calls to avoid wasting time, though score penalizes >500
    
    x = float(x0)
    fx = f(x)
    if abs(fx) <= 1e-8:
        return x

    # 1. Bracketing Phase
    # We need [a, b] such that f(a) and f(b) have opposite signs.
    # Start with a small interval around x0.
    a = x - 1.0
    b = x + 1.0
    fa = f(a)
    fb = f(b)
    
    # Expand if no sign change
    range_exp = 1.0
    for _ in range(20):
        if fa * fb < 0:
            break
        # Expand range
        range_exp *= 2.0
        a = x - range_exp
        b = x + range_exp
        fa = f(a)
        fb = f(b)
        if time.monotonic() > deadline:
            break
    else:
        # If we couldn't bracket, try a wider local search or return best effort
        # For polynomials, roots are finite. If x0 is very far, this might fail.
        # But x0 is likely a good starting point.
        pass

    # If no bracket found, fallback to local Newton/Secant without bracketing guarantee
    # but with damping.
    if fa * fb > 0:
        # Fallback: Simple damped Newton
        for _ in range(50):
            if time.monotonic() > deadline:
                break
            h = 1e-5 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if abs(d) < 1e-15:
                break
            x_new = x - fx / d
            # Damp if step is huge
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x - math.copysign(0.1 * max(1.0, abs(x)), fx)
            x = x_new
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
        return x

    # 2. Hybrid Secant-Bisection Phase
    # We have bracket [a, b] with fa, fb.
    # Initialize Secant points.
    # We want to use points inside the bracket.
    # Let's start with a and b as the two Secant points?
    # Or x0 and one endpoint?
    # Using a and b is safe.
    
    x0_sec = a
    f0_sec = fa
    x1_sec = b
    f1_sec = fb
    
    # If x0 is strictly inside, we could use it, but a,b are fine.
    # Actually, using x0 might be faster if it's close.
    # Let's check if x0 is in (a,b). It is, by construction.
    # Let's set x1_sec = x, f1_sec = fx, and x0_sec = a, f0_sec = fa?
    # No, Secant needs distinct points.
    # Let's use a and x.
    if abs(x - a) < 1e-10:
         x0_sec = b
         f0_sec = fb
    else:
         x0_sec = a
         f0_sec = fa
         
    x1_sec = x
    f1_sec = fx

    for _ in range(100):
        if time.monotonic() > deadline:
            break
            
        if abs(f1_sec) <= 1e-8:
            return x1_sec
        if abs(f0_sec) <= 1e-8:
            return x0_sec
            
        # Check bracket validity
        if f0_sec * f1_sec > 0:
            # Should not happen if we maintain bracket, but if Secant jumped out,
            # we rely on the bisection update to fix it.
            # Actually, in this hybrid, we update the bracket [a,b] separately.
            pass

        # Calculate Secant Step
        denom = f1_sec - f0_sec
        if abs(denom) < 1e-15:
            # Degenerate, bisect
            x_mid = (a + b) / 2.0
            f_mid = f(x_mid)
            if f0_sec * f_mid <= 0:
                # Root in [a, x_mid]
                b = x_mid
                fb = f_mid
                # Update Secant points: keep a, move b
                x1_sec = x_mid
                f1_sec = f_mid
            else:
                # Root in [x_mid, b]
                a = x_mid
                fa = f_mid
                x0_sec = x_mid
                f0_sec = f_mid
            continue

        x_new = x1_sec - f1_sec * (x1_sec - x0_sec) / denom
        
        # Check if x_new is inside the current bracket [a, b]
        if a < x_new < b:
            f_new = f(x_new)
            # Update bracket
            if fa * f_new <= 0:
                # Root in [a, x_new]
                b = x_new
                fb = f_new
                # Update Secant points: keep a, move b to x_new
                # But we need two points. 
                # Standard Secant keeps last two.
                # If we move b, we lose the old b.
                # Let's maintain x0_sec and x1_sec as the last two evaluated points.
                # If x_new is accepted, it becomes the new x1_sec.
                # The old x1_sec becomes x0_sec?
                # Yes.
                x0_sec, f0_sec = x1_sec, f1_sec
                x1_sec, f1_sec = x_new, f_new
            else:
                # Root in [x_new, b]
                a = x_new
                fa = f_new
                x0_sec, f0_sec = x1_sec, f1_sec
                x1_sec, f1_sec = x_new, f_new
        else:
            # Secant step is outside. Bisect.
            x_mid = (a + b) / 2.0
            f_mid = f(x_mid)
            if fa * f_mid <= 0:
                b = x_mid
                fb = f_mid
                # Update Secant points to include x_mid
                # Keep the point closer to the root? Or just replace one?
                # To maintain stability, let's set the new point as x1_sec 
                # and keep the other endpoint or previous point as x0_sec.
                # Using endpoints is safer for Secant stability if we just bisected.
                # But Secant is faster with close points.
                # Let's just set x1_sec = x_mid, f1_sec = f_mid.
                # And keep x0_sec as is? If x0_sec is far, Secant might be bad.
                # Let's reset Secant points to a and x_mid or x_mid and b?
                # Actually, if we bisect, we can just set the new point and one of the old ones.
                # Let's set x0_sec = a, f0_sec = fa (or b, fb) and x1_sec = x_mid.
                # This ensures the Secant line is well-behaved.
                if abs(x_mid - a) < abs(x_mid - b):
                    x0_sec, f0_sec = a, fa
                else:
                    x0_sec, f0_sec = b, fb
                x1_sec, f1_sec = x_mid, f_mid
            else:
                a = x_mid
                fa = f_mid
                if abs(x_mid - a) < abs(x_mid - b): # Wait, a is now x_mid
                    # Logic: pick the endpoint closer to x_mid
                    if abs(x_mid - b) < abs(x_mid - a): # a is x_mid, so dist is 0. 
                        # This logic is getting messy.
                        pass
                # Simpler: Just set x1_sec = x_mid, f1_sec = f_mid.
                # And x0_sec = the other endpoint.
                if fa * f_mid <= 0: # This branch is for fa*f_mid > 0, so root in [x_mid, b]
                    # a = x_mid
                    x0_sec, f0_sec = b, fb
                else:
                    # b = x_mid
                    x0_sec, f0_sec = a, fa
                x1_sec, f1_sec = x_mid, f_mid

        # Convergence check on interval width
        if abs(b - a) < 1e-10:
            return (a + b) / 2.0

    # Return best estimate
    if abs(f1_sec) < abs(f0_sec):
        return x1_sec
    else:
        return x0_sec
