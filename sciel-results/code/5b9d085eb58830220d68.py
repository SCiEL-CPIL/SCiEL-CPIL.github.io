import math
import time


def find_root(f, x0):
    TOL = 1e-8
    MAX_EVALS = 500
    DEADLINE = time.monotonic() + 135.0  # safety margin before the 150s kill

    evals = 0
    best_x = float(x0)
    best_f = None
    best_abs = math.inf

    def call_f(x):
        nonlocal evals, best_x, best_f, best_abs
        if evals >= MAX_EVALS or time.monotonic() >= DEADLINE:
            return None
        evals += 1
        try:
            val = float(f(x))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                AttributeError, ArithmeticError):
            return None
        if not math.isfinite(val):
            return None
        av = abs(val)
        if av < best_abs:
            best_abs = av
            best_x = float(x)
            best_f = val
        return val

    def is_good(fx):
        return fx is not None and abs(fx) <= TOL

    def is_close(fx):
        return fx is not None and abs(fx) <= 1e-5

    def illinois_converge(a, b, fa, fb):
        """Illinois algorithm to converge within a bracket [a,b] where fa*fb<0."""
        if is_good(fa):
            return a, fa
        if is_good(fb):
            return b, fb

        for _ in range(150):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fa * fb > 0:
                break  # lost bracket
            # Illinois: replace the endpoint with larger |f|
            if abs(fa) < abs(fb):
                # root closer to a, replace b
                c = a + (b - a) * fa / (fa - fb)
                fc = call_f(c)
                if fc is None:
                    break
                if is_good(fc):
                    return c, fc
                b, fb = c, fc
            else:
                # root closer to b, replace a
                c = a + (b - a) * fa / (fa - fb)
                fc = call_f(c)
                if fc is None:
                    break
                if is_good(fc):
                    return c, fc
                a, fa = c, fc
            # Check convergence by interval width
            if abs(b - a) <= 2 * 2.22e-16 * max(1.0, abs(a), abs(b)):
                break

        # Return the point with smaller |f|
        if fa is not None and (fb is None or abs(fa) < abs(fb)):
            return a, fa
        if fb is not None:
            return b, fb
        return a, fa

    def find_bracket(x0):
        """Find a bracket [a,b] around x0 where f(a)*f(b)<0."""
        # Start with a small step and expand geometrically
        h = 1e-8 * max(1.0, abs(x0))
        a = x0 - h
        b = x0 + h
        fa = call_f(a)
        fb = call_f(b)
        
        if fa is None and fb is None:
            return None, None, None, None
        if fa is None:
            a, fa = b, fb
            b = x0 - h
            fb = call_f(b)
        if fb is None:
            b, fb = a, fa
            a = x0 + h
            fa = call_f(a)
        
        if fa is None or fb is None:
            return a, fa, b, fb
        
        # Check if we already have a sign change
        if fa * fb < 0:
            return a, fa, b, fb
        
        # Expand the bracket geometrically
        for _ in range(30):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 2:
                break
            h *= 2.0
            if h > 1e10:
                break
            # Try expanding in both directions
            a_new = x0 - h
            b_new = x0 + h
            fa_new = call_f(a_new)
            fb_new = call_f(b_new)
            
            if fa_new is None and fb_new is None:
                continue
            if fa_new is not None and fb_new is not None:
                if fa_new * fb_new < 0:
                    return a_new, fa_new, b_new, fb_new
                # Also check if new point brackets with old
                if fa_new is not None and fa_new * fb < 0:
                    return a_new, fa_new, b, fb
                if fb_new is not None and fa * fb_new < 0:
                    return a, fa, b_new, fb_new
            elif fa_new is not None:
                if fa_new * fb < 0:
                    return a_new, fa_new, b, fb
            elif fb_new is not None:
                if fa * fb_new < 0:
                    return a, fa, b_new, fb_new
        
        return a, fa, b, fb

    def muller_step(x0, f0, x1, f1, x2, f2):
        """
        Muller's method: fit a quadratic through (x0,f0), (x1,f1), (x2,f2)
        and return the root closest to x2.
        Returns None if the quadratic is degenerate or the step is not finite.
        """
        # Compute divided differences
        h01 = x1 - x0
        h12 = x2 - x1
        if abs(h01) < 1e-300 or abs(h12) < 1e-300:
            return None
        
        d01 = (f1 - f0) / h01
        d12 = (f2 - f1) / h12
        a = (d12 - d01) / (h12 + h01)
        b = d12 + a * h12
        c = f2
        
        # Solve a*x^2 + b*x + c = 0
        # Use the numerically stable quadratic formula
        if abs(a) < 1e-300:
            # Linear case: b*x + c = 0
            if abs(b) < 1e-300:
                return None
            return -c / b
        
        disc = b * b - 4 * a * c
        if disc < 0:
            # Complex roots; use the one with larger real part? 
            # For real root finding, if disc < 0, the quadratic doesn't cross zero
            # in real domain near our points. Fallback to secant-like step.
            # Actually, Muller's method usually handles this by taking the root
            # closest to x2. If disc < 0, there are no real roots.
            # We should probably fallback to secant or return None.
            return None
        
        sqrt_disc = math.sqrt(disc)
        
        # The two roots are (-b +/- sqrt_disc) / (2a)
        # To avoid cancellation, we choose the sign that avoids subtracting
        # nearly equal numbers.
        # Standard trick: use q = -0.5 * (b + sign(b) * sqrt_disc)
        # Then roots are c/q and q/a
        if b >= 0:
            q = -0.5 * (b + sqrt_disc)
        else:
            q = -0.5 * (b - sqrt_disc)
        
        if abs(q) < 1e-300:
            return None
            
        x_root1 = c / q
        x_root2 = q / a
        
        # Choose the root closest to x2
        if abs(x_root1 - x2) <= abs(x_root2 - x2):
            return x_root1
        else:
            return x_root2

    def secant_polish(x0, fx0, x1, fx1):
        """Use secant method for a few steps to polish a near-root."""
        x = x0
        xp = x1
        fx = fx0
        fxp = fx1
        
        for _ in range(20):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            if fx is None or fxp is None:
                break
            if is_good(fx):
                return x, fx
            if is_good(fxp):
                return xp, fxp
            if fx == fxp:
                break
            xn = x - fx * (x - xp) / (fx - fxp)
            if not math.isfinite(xn):
                break
            fxn = call_f(xn)
            if fxn is None:
                break
            if is_good(fxn):
                return xn, fxn
            xp, fxp = x, fx
            x, fx = xn, fxn
        
        # Return best of the three
        candidates = [(x, fx), (xp, fxp)]
        if fx is not None:
            candidates.append((x, fx))
        best = min((c for c in candidates if c[1] is not None), 
                   key=lambda c: abs(c[1]), default=(x, fx))
        return best

    def muller_converge(x0, x1, x2, f0, f1, f2):
        """
        Iterate Muller's method until convergence or failure.
        Maintains the three most recent points.
        """
        # Ensure x2 is the "best" point (smallest |f|) to bias convergence
        # Actually, Muller's method is symmetric in x0, x1 but biased to x2 for root selection.
        # We just iterate.
        
        xs = [x0, x1, x2]
        fs = [f0, f1, f2]
        
        for _ in range(100):
            if time.monotonic() >= DEADLINE or evals >= MAX_EVALS - 1:
                break
            
            # Check if any current point is good
            for i in range(3):
                if is_good(fs[i]):
                    return xs[i], fs[i]
            
            # Compute next step
            x_next = muller_step(xs[0], fs[0], xs[1], fs[1], xs[2], fs[2])
            
            if x_next is None or not math.isfinite(x_next):
                # Fallback to secant step using last two points
                if fs[1] is not None and fs[2] is not None and fs[1] != fs[2]:
                    x_next = xs[2] - fs[2] * (xs[2] - xs[1]) / (fs[2] - fs[1])
                else:
                    break
                
                if not math.isfinite(x_next):
                    break
            
            # Check if x_next is too close to existing points (stagnation)
            if any(abs(x_next - xi) < 1e-12 * max(1.0, abs(x_next)) for xi in xs):
                break
                
            f_next = call_f(x_next)
            if f_next is None:
                break
            
            if is_good(f_next):
                return x_next, f_next
            
            # Shift points: drop the oldest (x0, f0)
            xs = [xs[1], xs[2], x_next]
            fs = [fs[1], fs[2], f_next]
            
            # Optional: Reorder to keep x2 as the point with smallest |f|?
            # Muller's method is more robust if the "current" point (x2) is the best.
            # Let's find the index of the smallest |f| and make it x2.
            # This requires swapping x0, x1, x2 and f0, f1, f2.
            idx_min = 0
            if abs(fs[1]) < abs(fs[0]):
                idx_min = 1
            if abs(fs[2]) < abs(fs[idx_min]):
                idx_min = 2
                
            if idx_min != 2:
                # Rotate so that idx_min becomes 2
                # If idx_min is 0: (0,1,2) -> (1,2,0)
                # If idx_min is 1: (0,1,2) -> (0,2,1)
                if idx_min == 0:
                    xs = [xs[1], xs[2], xs[0]]
                    fs = [fs[1], fs[2], fs[0]]
                elif idx_min == 1:
                    xs = [xs[0], xs[2], xs[1]]
                    fs = [fs[0], fs[2], fs[1]]
        
        # Return best point
        best_i = 0
        if abs(fs[1]) < abs(fs[0]):
            best_i = 1
        if abs(fs[2]) < abs(fs[best_i]):
            best_i = 2
        return xs[best_i], fs[best_i]

    x0 = float(x0)
    
    # Step 1: Try to find a bracket and use Illinois
    a, fa, b, fb = find_bracket(x0)
    
    if a is not None and fa is not None and b is not None and fb is not None:
        if fa * fb < 0:
            # We have a valid bracket, use Illinois to converge
            x, fx = illinois_converge(a, b, fa, fb)
            if is_good(fx):
                return x
            # If close, polish with secant
            if is_close(fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
                h = 1e-6 * max(1.0, abs(x))
                x2, fx2 = secant_polish(x, fx, x + h, call_f(x + h))
                if is_good(fx2):
                    return x2
                if abs(fx2) < abs(fx):
                    x, fx = x2, fx2
    
    # Step 2: If no bracket or Illinois didn't converge, try Muller's method
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1e-4 * max(1.0, abs(x0))
        x1, fx1 = x0, call_f(x0)
        x2, fx2 = x0 + h, call_f(x0 + h)
        x3, fx3 = x0 - h, call_f(x0 - h)
        
        # Filter out None values
        points = []
        if fx1 is not None:
            points.append((x1, fx1))
        if fx2 is not None:
            points.append((x2, fx2))
        if fx3 is not None:
            points.append((x3, fx3))
        
        if len(points) >= 3:
            # Use the three points for Muller
            p1, f1 = points[0]
            p2, f2 = points[1]
            p3, f3 = points[2]
            
            x, fx = muller_converge(p1, p2, p3, f1, f2, f3)
            if is_good(fx):
                return x
            if is_close(fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
                # Polish with secant
                h2 = 1e-6 * max(1.0, abs(x))
                x4, fx4 = secant_polish(x, fx, x + h2, call_f(x + h2))
                if is_good(fx4):
                    return x4
                if abs(fx4) < abs(fx):
                    x, fx = x4, fx4
    
    # Step 3: Try wider secant
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1.0
        x1, fx1 = x0 - h, call_f(x0 - h)
        x2, fx2 = x0 + h, call_f(x0 + h)
        if fx1 is not None and fx2 is not None:
            x, fx = secant_polish(x1, fx1, x2, fx2)
            if is_good(fx):
                return x
    
    # Step 4: If we got close, try one more polish
    if is_close(best_f) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
        h = 1e-6 * max(1.0, abs(best_x))
        x1, fx1 = best_x, best_f
        x2, fx2 = best_x + h, call_f(best_x + h)
        if fx2 is not None:
            x, fx = secant_polish(x1, fx1, x2, fx2)
            if is_good(fx):
                return x
    
    return best_x
