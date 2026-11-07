import math
import time
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    
    Strategy:
    1. Check if x0 is already a root.
    2. Attempt to reconstruct the polynomial via Lagrange interpolation using 5 points.
       Since the task specifies "3 fixed smooth polynomial equations", and typically these
       are low-degree (likely cubic or lower), 5 points should be enough to reconstruct
       the polynomial exactly (or nearly so) if the degree is <= 4.
       Then find roots of the reconstructed polynomial analytically/numerically.
       This reduces the number of calls to ~5-6 per equation if successful.
    3. If interpolation fails or doesn't yield a valid root, fall back to a robust
       safeguarded Newton/bisection hybrid with a deadline guard.
    """
    TOL = 1e-8
    DEADLINE = 110.0  # Safety margin before 150s kill
    START_TIME = time.time()
    
    def is_finite(x):
        return math.isfinite(x)

    def check_deadline():
        return (time.time() - START_TIME) < DEADLINE

    # Initial check
    try:
        fx0 = f(x0)
    except Exception:
        return x0
        
    if not is_finite(fx0):
        return x0

    if abs(fx0) <= TOL:
        return x0

    best_x = x0
    best_f_abs = abs(fx0)

    # --- Strategy 1: Polynomial Reconstruction via Lagrange Interpolation ---
    # We assume f is a polynomial of degree <= 4.
    # We pick 5 points around x0 with a small spread to avoid numerical instability.
    # The spread should be small enough that the polynomial doesn't vary wildly,
    # but large enough to be distinct.
    
    # Try a few different spreads to be robust against x0 being far from the root
    # or the polynomial having large coefficients.
    spreads = [1.0, 0.1, 10.0, 0.01, 100.0]
    
    for spread in spreads:
        if not check_deadline():
            break
            
        # Pick 5 points: x0 - 2*spread, x0 - spread, x0, x0 + spread, x0 + 2*spread
        # Using x0 itself saves one call if we already have fx0.
        points = [x0 - 2*spread, x0 - spread, x0, x0 + spread, x0 + 2*spread]
        
        # We already have f(x0). Let's get the others.
        try:
            f_vals = [None] * 5
            f_vals[2] = fx0
            
            for i in range(5):
                if i == 2:
                    continue
                val = f(points[i])
                if not is_finite(val):
                    break
                f_vals[i] = val
            else:
                # All 5 values obtained successfully
                # Check if any is already a root
                for i in range(5):
                    if abs(f_vals[i]) <= TOL:
                        return points[i]
                    if abs(f_vals[i]) < best_f_abs:
                        best_f_abs = abs(f_vals[i])
                        best_x = points[i]
                
                # Try to reconstruct the polynomial coefficients using numpy.polyfit
                # We use degree 4 to be safe, but if the actual degree is lower,
                # the higher coefficients should be near zero.
                try:
                    coeffs = np.polyfit(points, f_vals, 4)
                    # Find roots of the reconstructed polynomial
                    roots = np.roots(coeffs)
                    
                    # Filter for real roots
                    real_roots = []
                    for r in roots:
                        if abs(r.imag) < 1e-10:
                            real_roots.append(r.real)
                    
                    # For each real root, check if it's a valid root of f
                    # We pick the one closest to x0 to be safe, or the one with smallest |f(r)|
                    candidate_root = None
                    min_f_abs = float('inf')
                    
                    for r in real_roots:
                        if not is_finite(r):
                            continue
                        # Check if this root is "close" to our interpolation points
                        # to ensure it's not an artifact of high-degree fitting noise
                        # Actually, for a true polynomial, it should be exact.
                        # Let's just evaluate f(r) to verify.
                        try:
                            fr = f(r)
                        except Exception:
                            continue
                        if not is_finite(fr):
                            continue
                        if abs(fr) < min_f_abs:
                            min_f_abs = abs(fr)
                            candidate_root = r
                        if abs(fr) <= TOL:
                            candidate_root = r
                            break
                        if abs(fr) < best_f_abs:
                            best_f_abs = abs(fr)
                            best_x = r
                    
                    if candidate_root is not None and is_finite(candidate_root):
                        # Verify once more
                        try:
                            f_check = f(candidate_root)
                            if is_finite(f_check) and abs(f_check) <= TOL:
                                return candidate_root
                        except Exception:
                            pass
                    # If no root found within tolerance, we might still have a good candidate
                    # but we should probably try other spreads or fall back.
                    # However, if the interpolation was accurate, the root should be accurate.
                    # If |f(r)| is small but > TOL, we can use Newton to polish.
                    if candidate_root is not None and is_finite(candidate_root) and min_f_abs < 1e-4:
                        # Polish with Newton
                        x = candidate_root
                        fx = f(x)
                        for _ in range(20):
                            if not check_deadline():
                                break
                            if abs(fx) <= TOL:
                                return x
                            h = 1e-6 * max(1.0, abs(x))
                            try:
                                fx_h = f(x + h)
                                fx_mh = f(x - h)
                            except Exception:
                                break
                            if not (is_finite(fx_h) and is_finite(fx_mh)):
                                break
                            dfx = (fx_h - fx_mh) / (2 * h)
                            if abs(dfx) < 1e-15:
                                break
                            step = fx / dfx
                            x_new = x - step
                            if not is_finite(x_new):
                                break
                            try:
                                fx_new = f(x_new)
                            except Exception:
                                break
                            if not is_finite(fx_new):
                                break
                            if abs(fx_new) < abs(fx):
                                x = x_new
                                fx = fx_new
                                if abs(fx) <= TOL:
                                    return x
                            else:
                                break
                        if abs(fx) <= TOL:
                            return x
                            
                except Exception:
                    pass
        except Exception:
            continue

    # --- Strategy 2: Fallback Safeguarded Newton/Bisection ---
    # If interpolation failed, use a robust method.
    # First, try to find a bracket.
    step = 1.0
    a = None
    b = None
    fa = None
    fb = None
    
    for i in range(50):
        if not check_deadline():
            break
        a = x0 - step
        b = x0 + step
        try:
            fa = f(a)
            fb = f(b)
        except Exception:
            break
        if not (is_finite(fa) and is_finite(fb)):
            break
        
        if abs(fa) < best_f_abs:
            best_f_abs = abs(fa)
            best_x = a
        if abs(fb) < best_f_abs:
            best_f_abs = abs(fb)
            best_x = b
            
        if fa * fb < 0:
            # Bracket found
            try:
                import scipy.optimize
                root = scipy.optimize.brentq(f, a, b, xtol=1e-12, rtol=1e-12, maxiter=100)
                if is_finite(root):
                    f_root = f(root)
                    if is_finite(f_root) and abs(f_root) <= TOL:
                        return root
                    return root
            except Exception:
                break
        step *= 2.0
        if step > 1e10:
            break

    # If no bracket, use Newton
    x = x0
    fx = fx0
    for i in range(50):
        if not check_deadline():
            break
        if abs(fx) <= TOL:
            return x
        h = 1e-5 * max(1.0, abs(x))
        try:
            fx_h = f(x + h)
            fx_mh = f(x - h)
        except Exception:
            break
        if not (is_finite(fx_h) and is_finite(fx_mh)):
            break
        dfx = (fx_h - fx_mh) / (2 * h)
        if abs(dfx) < 1e-15:
            break
        step = fx / dfx
        x_new = x - step
        if not is_finite(x_new):
            break
        try:
            fx_new = f(x_new)
        except Exception:
            break
        if not is_finite(fx_new):
            break
        if abs(fx_new) < best_f_abs:
            best_f_abs = abs(fx_new)
            best_x = x_new
        if abs(fx_new) <= TOL:
            return x_new
        if abs(fx_new) > abs(fx) * 1.1 and i > 2:
            break
        x = x_new
        fx = fx_new

    return best_x
# EVOLVE-BLOCK-END
