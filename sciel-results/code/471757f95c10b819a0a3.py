"""Single-shot cubic reconstruction via adaptive point selection."""
import numpy as np

def find_root(f, x0):
    x = float(x0)
    
    # Step 1: Check if x0 is already the root
    try:
        y0 = f(x)
    except Exception:
        return x
        
    if abs(y0) <= 1e-8:
        return x

    # Step 2: Set up interpolation points
    # Use a small step size relative to x
    h = 1e-3 * max(1.0, abs(x))
    
    # Points: x-h, x, x+h, x+2h
    # This asymmetric set often better captures local curvature if root is nearby
    p1 = x - h
    p2 = x
    p3 = x + h
    p4 = x + 2 * h
    
    y1 = y0 # f(x) already computed
    
    # Evaluate remaining 3 points
    try:
        y_p1 = f(p1)
        y_p3 = f(p3)
        y_p4 = f(p4)
    except Exception:
        # If evaluation fails, fallback to Newton
        return _newton_fallback(f, x)

    # Step 3: Try Quadratic Fit first (3 points: p1, x, p3)
    # If the function is quadratic, this is exact.
    # Points: (p1, y_p1), (x, y0), (p3, y_p3)
    try:
        # Lagrange or polyfit
        # Using np.polyfit for 2nd degree
        xs_q = np.array([p1, x, p3], dtype=np.float64)
        ys_q = np.array([y_p1, y0, y_p3], dtype=np.float64)
        coeffs_q = np.polyfit(xs_q, ys_q, 2)
        roots_q = np.roots(coeffs_q)
        
        # Filter real roots
        real_roots_q = [r.real for r in roots_q if abs(r.imag) < 1e-10]
        
        if real_roots_q:
            # Pick closest to x
            x_q = min(real_roots_q, key=lambda r: abs(r - x))
            y_q = f(x_q)
            if abs(y_q) <= 1e-8:
                return x_q
    except Exception:
        pass

    # Step 4: Try Cubic Fit (4 points: p1, x, p3, p4)
    try:
        xs_c = np.array([p1, x, p3, p4], dtype=np.float64)
        ys_c = np.array([y_p1, y0, y_p3, y_p4], dtype=np.float64)
        coeffs_c = np.polyfit(xs_c, ys_c, 3)
        roots_c = np.roots(coeffs_c)
        
        real_roots_c = [r.real for r in roots_c if abs(r.imag) < 1e-10]
        
        if real_roots_c:
            x_c = min(real_roots_c, key=lambda r: abs(r - x))
            y_c = f(x_c)
            if abs(y_c) <= 1e-8:
                return x_c
            
            # If cubic root is close but not quite, try one Newton step from x_c
            # This uses 1 extra call
            try:
                h_newton = 1e-6 * max(1.0, abs(x_c))
                d = (f(x_c + h_newton) - y_c) / h_newton
                if d != 0:
                    x_newton = x_c - y_c / d
                    y_newton = f(x_newton)
                    if abs(y_newton) <= 1e-8:
                        return x_newton
            except Exception:
                pass
    except Exception:
        pass

    # Step 5: Final Fallback Newton from x
    return _newton_fallback(f, x)

def _newton_fallback(f, x):
    for _ in range(30):
        try:
            fx = f(x)
            if abs(fx) <= 1e-8:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x = x - fx / d
        except Exception:
            break
    return x
