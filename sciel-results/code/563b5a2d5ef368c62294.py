"""Root finder for smooth polynomial equations with call-budget awareness."""
import math
import time
import numpy as np

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _find_polynomial_roots(coeffs):
    """Find real roots of polynomial given by coefficients (highest degree first)."""
    if len(coeffs) == 0:
        return []
    if len(coeffs) == 1:
        return []
    if len(coeffs) == 2:
        # Linear: a*x + b = 0
        a, b = coeffs
        if abs(a) < 1e-15:
            return []
        return [-b / a]
    # Use numpy to find roots
    try:
        roots = np.roots(coeffs)
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        return real_roots
    except Exception:
        return []


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8.
    Uses polynomial interpolation strategy to predict roots, then refines with Newton/bisection.
    """
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)
    
    # Track best solution found so far
    best_x = x
    best_f = f(x)
    if not _is_finite(best_f):
        return 0.0
    if abs(best_f) <= 1e-8:
        return best_x

    # Phase 1: Gather points for polynomial interpolation
    # Start with x0, then expand to get more points
    points = [(x, best_f)]
    
    # Try to find a bracket first with a few evaluations
    a, b = x, None
    fa, fb = best_f, None
    
    # Expand in both directions to find sign change
    step = 1.0
    for direction in [1, -1]:
        if b is not None:
            break
        for _ in range(20):
            if _time_left() < 10.0:
                break
            x_try = x + direction * step
            f_try = f(x_try)
            if not _is_finite(f_try):
                step *= 2.0
                continue
            if abs(f_try) <= 1e-8:
                return x_try
            if fa * f_try < 0:
                if direction == 1:
                    a, b = x, x_try
                    fa, fb = best_f, f_try
                else:
                    a, b = x_try, x
                    fa, fb = f_try, best_f
                break
            step *= 2.0
            if step > 1e10:
                break
        if b is not None:
            break

    # If we found a bracket, use bisection/Brent-like refinement
    if b is not None:
        # Use inverse quadratic interpolation or bisection to refine
        # Start with the bracket
        lo, hi = a, b
        f_lo, f_hi = fa, fb
        
        # Use secant method within the bracket
        x_prev, x_curr = lo, hi
        f_prev, f_curr = f_lo, f_hi
        
        for _ in range(100):
            if _time_left() < 5.0:
                break
            if abs(f_curr) < abs(f_prev):
                x_prev, x_curr = x_curr, x_prev
                f_prev, f_curr = f_curr, f_prev
            
            # Secant step
            if f_curr == f_prev:
                break
            x_new = x_curr - f_curr * (x_curr - x_prev) / (f_curr - f_prev)
            
            # If secant step is outside bracket, use bisection
            if not (lo < x_new < hi):
                x_new = (lo + hi) / 2.0
            
            f_new = f(x_new)
            if not _is_finite(f_new):
                break
            
            if abs(f_new) <= 1e-8:
                return x_new
            
            # Update bracket
            if f_new * f_lo < 0:
                hi, f_hi = x_new, f_new
            else:
                lo, f_lo = x_new, f_new
            
            x_prev, x_curr = x_curr, x_new
            f_prev, f_curr = f_curr, f_new
            
            if abs(hi - lo) < 1e-12 * max(1.0, abs(x_new)):
                break
        
        # Return the point with smallest |f|
        candidates = [x_curr, (lo + hi) / 2.0]
        for cand in candidates:
            if _is_finite(cand):
                f_cand = f(cand)
                if _is_finite(f_cand) and abs(f_cand) < abs(best_f):
                    best_x, best_f = cand, f_cand
        
        if abs(best_f) <= 1e-8:
            return best_x
    
    # Phase 2: No bracket found. Use polynomial interpolation.
    # Gather more points around x0
    if _time_left() > 15.0:
        # Try to fit a polynomial using points near x0
        # Use points: x0, x0+1, x0-1, x0+2, x0-2
        interp_points = [(x, best_f)]
        offsets = [1.0, -1.0, 2.0, -2.0, 4.0, -4.0]
        for off in offsets:
            if _time_left() < 10.0:
                break
            x_p = x + off
            f_p = f(x_p)
            if _is_finite(f_p):
                interp_points.append((x_p, f_p))
                if abs(f_p) <= 1e-8:
                    return x_p
                if len(interp_points) >= 5:
                    break
        
        # Try to fit polynomial of degree <= 4 and find its roots
        if len(interp_points) >= 3:
            xs = [p[0] for p in interp_points]
            ys = [p[1] for p in interp_points]
            n = len(xs)
            deg = min(n - 1, 4)
            
            # Fit polynomial using numpy
            try:
                coeffs = np.polyfit(xs, ys, deg)
                real_roots = _find_polynomial_roots(coeffs)
                
                # Evaluate each real root and pick the best
                for root in real_roots:
                    if _time_left() < 5.0:
                        break
                    if not _is_finite(root):
                        continue
                    # Only consider roots near our search region
                    if abs(root - x) > 1e6:
                        continue
                    f_root = f(root)
                    if _is_finite(f_root):
                        if abs(f_root) < abs(best_f):
                            best_x, best_f = root, f_root
                        if abs(f_root) <= 1e-8:
                            return root
            except Exception:
                pass
    
    # Phase 3: If still no good root, try Newton from best_x
    if _time_left() > 10.0 and abs(best_f) > 1e-8:
        x_curr = best_x
        f_curr = best_f
        
        for _ in range(30):
            if _time_left() < 5.0:
                break
            h = 1e-8 * max(1.0, abs(x_curr))
            f_plus = f(x_curr + h)
            if not _is_finite(f_plus):
                break
            df = (f_plus - f_curr) / h
            if df == 0 or not _is_finite(df):
                break
            
            x_new = x_curr - f_curr / df
            if not _is_finite(x_new):
                break
            
            if abs(x_new - x_curr) > 10 * max(1.0, abs(x_curr)):
                x_new = x_curr + 0.5 * (x_new - x_curr)
            
            f_new = f(x_new)
            if not _is_finite(f_new):
                break
            
            if abs(f_new) < abs(f_curr):
                x_curr = x_new
                f_curr = f_new
                if abs(f_curr) < abs(best_f):
                    best_x, best_f = x_curr, f_curr
                if abs(f_curr) <= 1e-8:
                    return x_curr
            else:
                break
    
    return best_x
