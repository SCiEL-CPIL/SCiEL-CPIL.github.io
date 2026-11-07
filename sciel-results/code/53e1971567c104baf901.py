"""Combined root finder using Inverse Quadratic Interpolation (IQI) for speed and robust fallbacks."""
import math
import time

# Global deadline: evaluator kills after 150s, we use 140s as a safety margin
_DEADLINE = time.time() + 140.0
_CALL_BUDGET = 480  # Stay well under 500 per equation


def _time_left():
    return _DEADLINE - time.time()


def _safe_call(f, x, counter):
    """Call f(x) safely, returning nan on failure. Increments counter."""
    try:
        val = f(x)
        counter[0] += 1
        if not math.isfinite(val):
            return float('nan')
        return float(val)
    except Exception:
        counter[0] += 1
        return float('nan')


def _iqi_step(f, x0, fx0, x1, fx1, x2, fx2, counter):
    """
    Perform one step of Inverse Quadratic Interpolation.
    Returns (x_new, fx_new).
    Assumes x0, x1, x2 are distinct.
    """
    # Inverse quadratic interpolation:
    # We model x as a function of y (inverse of f).
    # x(y) = x0 * (y-x1)(y-x2)/((x0-x1)(x0-x2)) + ...
    
    # To avoid division by zero or instability, we check if points are too close
    if abs(x1 - x0) < 1e-15 * max(1.0, abs(x0)) or \
       abs(x2 - x0) < 1e-15 * max(1.0, abs(x0)) or \
       abs(x2 - x1) < 1e-15 * max(1.0, abs(x1)):
        # Fallback to secant using the two closest points? Or just return nan to trigger fallback
        return float('nan'), float('nan')

    # Lagrange basis polynomials for x(y)
    # Let y = 0 be the target root value.
    # We want x(0).
    
    # Terms:
    # L0(0) = (0-x1)(0-x2) / ((x0-x1)(x0-x2))
    # L1(0) = (0-x0)(0-x2) / ((x1-x0)(x1-x2))
    # L2(0) = (0-x0)(0-x1) / ((x2-x0)(x2-x1))
    
    # Note: The standard formula for IQI step x_new when seeking f(x)=0 is:
    # x_new = x0 * (fx1*fx2)/((fx0-fx1)*(fx0-fx2)) + 
    #         x1 * (fx0*fx2)/((fx1-fx0)*(fx1-fx2)) +
    #         x2 * (fx0*fx1)/((fx2-fx0)*(fx2-fx1))
    
    # Let's implement this direct formula which is more stable for root finding
    denom0 = (fx0 - fx1) * (fx0 - fx2)
    denom1 = (fx1 - fx0) * (fx1 - fx2)
    denom2 = (fx2 - fx0) * (fx2 - fx1)
    
    if abs(denom0) < 1e-30 or abs(denom1) < 1e-30 or abs(denom2) < 1e-30:
        return float('nan'), float('nan')

    term0 = x0 * (fx1 * fx2) / denom0
    term1 = x1 * (fx0 * fx2) / denom1
    term2 = x2 * (fx0 * fx1) / denom2
    
    x_new = term0 + term1 + term2
    
    if not math.isfinite(x_new):
        return float('nan'), float('nan')
        
    fx_new = _safe_call(f, x_new, counter)
    return x_new, fx_new


def _iqi_solver(f, x0, counter, max_iter=30):
    """
    IQI solver. Starts with 3 points, then iterates.
    Returns (x, fx, success).
    """
    # Initial points
    # Use a small step to get 3 distinct points
    h = 1e-4 * max(1.0, abs(x0))
    
    x_a = x0
    fx_a = _safe_call(f, x_a, counter)
    if math.isnan(fx_a):
        return x_a, fx_a, False
        
    if abs(fx_a) <= 1e-8:
        return x_a, fx_a, True

    x_b = x0 + h
    fx_b = _safe_call(f, x_b, counter)
    if math.isnan(fx_b):
        return x_a, fx_a, abs(fx_a) <= 1e-8
        
    if abs(fx_b) <= 1e-8:
        return x_b, fx_b, True

    x_c = x0 - h
    fx_c = _safe_call(f, x_c, counter)
    if math.isnan(fx_c):
        # If we fail to get 3 points, fallback to secant-like behavior with 2 points
        # For now, return best of a,b
        if abs(fx_a) < abs(fx_b):
            return x_a, fx_a, abs(fx_a) <= 1e-8
        return x_b, fx_b, abs(fx_b) <= 1e-8

    if abs(fx_c) <= 1e-8:
        return x_c, fx_c, True

    # Keep track of the 3 most recent points
    # We want to ensure we don't have duplicate x values
    # Sort by x? No, IQI works with any 3 distinct points.
    # However, for stability, it's often good to keep the points close to the current best.
    
    # State: (x1, fx1), (x2, fx2), (x3, fx3)
    # Let's initialize with a, b, c
    x1, fx1 = x_a, fx_a
    x2, fx2 = x_b, fx_b
    x3, fx3 = x_c, fx_c
    
    best_x = x1
    best_fx = fx1
    if abs(fx2) < abs(best_fx):
        best_x, best_fx = x2, fx2
    if abs(fx3) < abs(best_fx):
        best_x, best_fx = x3, fx3

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
            
        # Attempt IQI step
        x_new, fx_new = _iqi_step(f, x1, fx1, x2, fx2, x3, fx3, counter)
        
        if math.isnan(fx_new):
            # IQI failed, try Secant step using the two points with smallest |f|?
            # Or just break and let outer logic handle it.
            # Let's try a simple secant step between the two closest points in x
            # Find two points with distinct x
            if abs(x1 - x2) > 1e-15 and abs(fx1 - fx2) > 1e-15:
                x_new = x2 - fx2 * (x2 - x1) / (fx2 - fx1)
                fx_new = _safe_call(f, x_new, counter)
            elif abs(x1 - x3) > 1e-15 and abs(fx1 - fx3) > 1e-15:
                x_new = x3 - fx3 * (x3 - x1) / (fx3 - fx1)
                fx_new = _safe_call(f, x_new, counter)
            else:
                break
                
            if math.isnan(fx_new):
                break

        if abs(fx_new) <= 1e-8:
            return x_new, fx_new, True

        # Update the set of 3 points
        # Strategy: Discard the point with the largest |f(x)| and add the new point.
        # This keeps the interpolation local to the best region.
        # Current points: (x1, fx1), (x2, fx2), (x3, fx3), (x_new, fx_new)
        
        # Find index of max |f| among old 3
        abs_f1 = abs(fx1)
        abs_f2 = abs(fx2)
        abs_f3 = abs(fx3)
        
        if abs_f1 >= abs_f2 and abs_f1 >= abs_f3:
            # Discard x1
            x1, fx1 = x_new, fx_new
        elif abs_f2 >= abs_f1 and abs_f2 >= abs_f3:
            # Discard x2
            x2, fx2 = x_new, fx_new
        else:
            # Discard x3
            x3, fx3 = x_new, fx_new
            
        # Update best
        if abs(fx_new) < abs(best_fx):
            best_x, best_fx = x_new, fx_new

    # Return best found
    return best_x, best_fx, abs(best_fx) <= 1e-8


def _bisection(f, a, b, fa, fb, counter, max_iter=80):
    """Bisection method. Returns (x, fx, success)."""
    if fa * fb > 0:
        # No bracket, return the better one
        if abs(fa) < abs(fb):
            return a, fa, abs(fa) <= 1e-8
        else:
            return b, fb, abs(fb) <= 1e-8
    
    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # No bracket
        c = (a + b) / 2.0
        fc = _safe_call(f, c, counter)
        if math.isnan(fc):
            break
        if abs(fc) <= 1e-8:
            return c, fc, True
        if fa * fc <= 0:
            b = c
            fb = fc
        else:
            a = c
            fa = fc
        if abs(b - a) < 1e-12 * max(1.0, abs(c)):
            break
    # Return the midpoint of the final bracket
    c = (a + b) / 2.0
    fc = _safe_call(f, c, counter)
    return c, fc, abs(fc) <= 1e-8


def _brentq_like(f, a, b, fa, fb, counter, max_iter=60):
    """
    Brent's method (simplified) for finding a root in [a, b] where fa*fb < 0.
    Returns (x, fx, success).
    """
    # Ensure fa < 0 and fb > 0 for consistency
    if fa > 0:
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d
    tol = 1e-12

    for _ in range(max_iter):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        if fa * fb > 0:
            break  # Lost bracket
        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(e) < tol:
            break  # Converged
        if abs(fc) < abs(fb):
            b = c
            fb = fc
            c = a
            fc = fa
            d = b - a
            e = d

        # Attempt interpolation
        if abs(d) < tol:
            break
        # Secant step
        m = 0.5 * (a - b)
        e2 = 2.0 * m
        p = 2.0 * m * (fc - fb)
        q = (fc - fa)
        if abs(p) >= abs(q):
            # Use secant
            s = p / q
        else:
            # Use quadratic interpolation
            s = m
            q = fc - fa
            r = fc - fb
            if abs(q - r) < 1e-30:
                s = m * (1.0 if p > 0 else -1.0)
            else:
                s = m * (p / (q - r))
        
        # Check if s is acceptable
        if ((3.0 * m - abs(e)) / 2.0 < s < abs(e) / 2.0) or \
           ((e == d and (s < 0 and fc < 0 or s > 0 and fc > 0)) or \
            (abs(s) >= abs(e) / 2.0)):
            s = m
            d = s
        else:
            e = d

        a = b
        fa = fb
        b = b + s
        fb = _safe_call(f, b, counter)
        if math.isnan(fb):
            break
        if abs(fb) <= 1e-8:
            return b, fb, True

    # Return best estimate
    return b, fb, abs(fb) <= 1e-8


def _find_bracket(f, x0, counter):
    """
    Try to find a bracket [a, b] around x0 where f(a)*f(b) < 0.
    Returns (a, b, fa, fb) or None.
    """
    fx0 = _safe_call(f, x0, counter)
    if math.isnan(fx0):
        return None
    if abs(fx0) <= 1e-8:
        return x0, x0, fx0, fx0

    # Expand outward to find a sign change
    step = 1.0
    a = x0
    fa = fx0

    for _ in range(30):
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        b = x0 + step
        fb = _safe_call(f, b, counter)
        if math.isnan(fb):
            break
        if fa * fb < 0:
            return a, b, fa, fb
        # Also check the other direction
        b2 = x0 - step
        fb2 = _safe_call(f, b2, counter)
        if math.isnan(fb2):
            break
        if fa * fb2 < 0:
            return b2, a, fb2, fa
        step *= 2.0

    return None


def find_root(f, x0):
    """
    Find a root x of f(x) = 0 with |f(x)| <= 1e-8.
    Returns x as a float.
    """
    counter = [0]
    x0 = float(x0)
    best_x = x0
    best_fx = _safe_call(f, x0, counter)
    best_abs = abs(best_fx) if not math.isnan(best_fx) else float('inf')

    if best_abs <= 1e-8:
        return x0

    # Strategy 1: Try IQI method first (fast, 1 call/iter after initial 3)
    x_iqi, fx_iqi, success = _iqi_solver(f, x0, counter, max_iter=30)
    if not math.isnan(fx_iqi):
        if abs(fx_iqi) < best_abs:
            best_x = x_iqi
            best_abs = abs(fx_iqi)
        if success:
            return x_iqi

    # Strategy 2: Try to find a bracket and use Brent's method
    bracket = _find_bracket(f, x0, counter)
    if bracket is not None:
        a, b, fa, fb = bracket
        if fa == fb and abs(fa) <= 1e-8:
            return a
        if fa * fb < 0:
            x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=50)
            if not math.isnan(fx_brent):
                if abs(fx_brent) < best_abs:
                    best_x = x_brent
                    best_abs = abs(fx_brent)
                if success:
                    return x_brent
            # Fallback to bisection if Brent didn't converge
            x_bis, fx_bis, success = _bisection(f, a, b, fa, fb, counter, max_iter=80)
            if not math.isnan(fx_bis):
                if abs(fx_bis) < best_abs:
                    best_x = x_bis
                    best_abs = abs(fx_bis)
                if success:
                    return x_bis

    # Strategy 3: Try more IQI iterations from the best point found
    if best_abs > 1e-8 and not math.isnan(best_fx):
        x_iqi2, fx_iqi2, success = _iqi_solver(f, best_x, counter, max_iter=25)
        if not math.isnan(fx_iqi2):
            if abs(fx_iqi2) < best_abs:
                best_x = x_iqi2
                best_abs = abs(fx_iqi2)
            if success:
                return x_iqi2

    # Strategy 4: Try a few more bracketing attempts with different starting points
    for offset in [0.1, -0.1, 0.5, -0.5, 2.0, -2.0]:
        if counter[0] >= _CALL_BUDGET or _time_left() < 0.5:
            break
        x_try = x0 + offset
        bracket = _find_bracket(f, x_try, counter)
        if bracket is not None:
            a, b, fa, fb = bracket
            if fa * fb < 0:
                x_brent, fx_brent, success = _brentq_like(f, a, b, fa, fb, counter, max_iter=40)
                if not math.isnan(fx_brent):
                    if abs(fx_brent) < best_abs:
                        best_x = x_brent
                        best_abs = abs(fx_brent)
                    if success:
                        return x_brent

    # Return the best result found
    return best_x
