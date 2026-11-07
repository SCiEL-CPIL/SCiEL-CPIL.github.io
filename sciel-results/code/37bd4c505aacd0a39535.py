"""Single-shot cubic reconstruction via Lagrange interpolation + numpy.roots."""
import numpy as np

# EVOLVE-BLOCK-START
def find_root(f, x0):
    # Evaluate f at 4 distinct points to reconstruct a cubic polynomial.
    # Choose points around x0 with a small spread to capture local behaviour.
    x = float(x0)
    
    # Spread: use a small base spacing, scale by magnitude of x0
    base_h = 1e-3 * max(1.0, abs(x))
    
    # Four distinct points
    points = [x - 1.5 * base_h, x - 0.5 * base_h, x + 0.5 * base_h, x + 1.5 * base_h]
    
    # Ensure distinctness (guard against floating-point issues)
    if len(set(points)) < 4:
        # Fallback: use slightly different offsets
        points = [x - 2.0 * base_h, x - 1.0 * base_h, x + 1.0 * base_h, x + 2.0 * base_h]
    
    y_vals = [f(p) for p in points]
    
    # Build cubic coefficients via Lagrange interpolation
    # For points x0, x1, x2, x3 with values y0, y1, y2, y3:
    # The cubic polynomial P(t) = sum_i y_i * L_i(t)
    # where L_i(t) = prod_{j != i} (t - x_j) / (x_i - x_j)
    
    xs = np.array(points, dtype=np.float64)
    ys = np.array(y_vals, dtype=np.float64)
    
    # Compute coefficients of the interpolating cubic
    # Use np.polyfit or manual Lagrange
    # np.polyfit returns highest degree first
    try:
        coeffs = np.polyfit(xs, ys, 3)  # [a3, a2, a1, a0]
    except Exception:
        # Fallback to Newton's method if interpolation fails
        x = float(x0)
        for _ in range(60):
            fx = f(x)
            if abs(fx) <= 1e-12:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x = x - fx / d
        return x
    
    # Find roots of the cubic
    try:
        roots = np.roots(coeffs)
    except Exception:
        # Fallback to Newton
        x = float(x0)
        for _ in range(60):
            fx = f(x)
            if abs(fx) <= 1e-12:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x = x - fx / d
        return x
    
    # Filter for real roots and pick the one nearest to x0
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
    
    if not real_roots:
        # No real roots found; fallback to Newton
        x = float(x0)
        for _ in range(60):
            fx = f(x)
            if abs(fx) <= 1e-12:
                return x
            h = 1e-6 * max(1.0, abs(x))
            d = (f(x + h) - fx) / h
            if d == 0:
                break
            x = x - fx / d
        return x
    
    # Pick the real root closest to x0
    best_root = min(real_roots, key=lambda r: abs(r - x))
    
    # Verify and refine if needed
    fx = f(best_root)
    if abs(fx) <= 1e-8:
        return float(best_root)
    
    # If the interpolated root isn't accurate enough, try a few more points
    # with a wider spread and re-interpolate
    for spread_factor in [5.0, 10.0, 20.0, 50.0, 100.0]:
        h2 = base_h * spread_factor
        points2 = [x - 1.5 * h2, x - 0.5 * h2, x + 0.5 * h2, x + 1.5 * h2]
        y_vals2 = [f(p) for p in points2]
        xs2 = np.array(points2, dtype=np.float64)
        ys2 = np.array(y_vals2, dtype=np.float64)
        
        try:
            coeffs2 = np.polyfit(xs2, ys2, 3)
            roots2 = np.roots(coeffs2)
            real_roots2 = [r.real for r in roots2 if abs(r.imag) < 1e-10]
            if real_roots2:
                best_root2 = min(real_roots2, key=lambda r: abs(r - x))
                fx2 = f(best_root2)
                if abs(fx2) <= 1e-8:
                    return float(best_root2)
        except Exception:
            continue
    
    # Final fallback: Newton's method
    x = float(x0)
    for _ in range(60):
        fx = f(x)
        if abs(fx) <= 1e-12:
            return x
        h = 1e-6 * max(1.0, abs(x))
        d = (f(x + h) - fx) / h
        if d == 0:
            break
        x = x - fx / d
    return x
# EVOLVE-BLOCK-END
