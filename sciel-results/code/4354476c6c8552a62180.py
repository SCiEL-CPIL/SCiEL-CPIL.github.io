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
    
    # Step 2: If no bracket or Illinois didn't converge, try secant from nearby points
    if time.monotonic() < DEADLINE and evals < MAX_EVALS - 5:
        h = 1e-4 * max(1.0, abs(x0))
        x1, fx1 = x0, call_f(x0)
        x2, fx2 = x0 + h, call_f(x0 + h)
        if fx1 is not None and fx2 is not None:
            x, fx = secant_polish(x1, fx1, x2, fx2)
            if is_good(fx):
                return x
            if is_close(fx) and time.monotonic() < DEADLINE and evals < MAX_EVALS - 10:
                h2 = 1e-6 * max(1.0, abs(x))
                x3, fx3 = secant_polish(x, fx, x + h2, call_f(x + h2))
                if is_good(fx3):
                    return x3
    
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
