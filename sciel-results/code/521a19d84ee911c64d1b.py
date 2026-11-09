"""Root finder for smooth polynomial equations with call-budget awareness."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0  # safety margin before 120s kill
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _newton_fd(f, x0, fx0=None, max_iter=8, tol=1e-8):
    """
    Fast Newton's method with finite-difference derivative.
    Returns (x, fx, calls) or None if it fails to converge quickly.
    If fx0 is provided, it is used as f(x0) without an extra call.
    """
    x = float(x0)
    calls = 0
    h = 1e-7 * max(1.0, abs(x))
    
    # Use provided fx0 or evaluate
    if fx0 is not None:
        fx = fx0
    else:
        fx = f(x)
        calls += 1
        if not _is_finite(fx):
            return None
    
    if abs(fx) <= tol:
        return (x, fx, calls)
    
    for i in range(max_iter):
        if _time_left() <= 0:
            return None
        
        # Finite difference derivative (forward difference)
        fx_plus = f(x + h)
        calls += 1
        if not _is_finite(fx_plus):
            return None
        
        fpx = (fx_plus - fx) / h
        if abs(fpx) < 1e-15:
            return None
        
        x_new = x - fx / fpx
        if not _is_finite(x_new):
            return None
        
        # Evaluate f at new point
        fx_new = f(x_new)
        calls += 1
        if not _is_finite(fx_new):
            return None
        
        if abs(fx_new) <= tol:
            return (x_new, fx_new, calls)
        
        if abs(x_new - x) < 1e-12 * max(1.0, abs(x)):
            return (x_new, fx_new, calls)
        
        x = x_new
        fx = fx_new
        h = 1e-7 * max(1.0, abs(x))
    
    return None


def _poly_root_cubic(a, b, c, d):
    """
    Find real roots of ax^3 + bx^2 + cx + d = 0.
    Returns list of real roots.
    """
    if abs(a) < 1e-15:
        # Quadratic or lower
        if abs(b) < 1e-15:
            if abs(c) < 1e-15:
                return []
            return [-d / c]
        # Quadratic: bx^2 + cx + d = 0
        disc = c * c - 4 * b * d
        if disc < -1e-15:
            return []
        elif abs(disc) < 1e-15:
            return [-c / (2 * b)]
        else:
            sq = math.sqrt(disc)
            return [(-c + sq) / (2 * b), (-c - sq) / (2 * b)]
    
    # Normalize
    b = b / a
    c = c / a
    d = d / a
    
    # Depressed cubic: t^3 + pt + q where x = t - b/3
    p = c - b * b / 3.0
    q = d + b * b * b / 27.0 - b * c / 3.0
    
    disc = (q / 2.0) ** 2 + (p / 3.0) ** 3
    
    roots = []
    if disc > 1e-15:
        # One real root
        sq = math.sqrt(disc)
        u = math.copysign(abs(-q / 2.0 + sq) ** (1/3), -q / 2.0 + sq)
        v = math.copysign(abs(-q / 2.0 - sq) ** (1/3), -q / 2.0 - sq)
        t1 = u + v
        roots.append(t1 - b / 3.0)
    elif abs(disc) < 1e-15:
        # Multiple roots
        if abs(q) < 1e-15 and abs(p) < 1e-15:
            roots.append(-b / 3.0)
        else:
            u = math.copysign(abs(-q / 2.0) ** (1/3), -q / 2.0)
            roots.append(2 * u - b / 3.0)
            roots.append(-u - b / 3.0)
    else:
        # Three real roots (trigonometric method)
        r = math.sqrt(-p ** 3 / 27.0)
        phi = math.acos(max(-1.0, min(1.0, -q / (2.0 * r))))
        m = 2.0 * math.sqrt(-p / 3.0)
        for k in range(3):
            t = m * math.cos(phi / 3.0 - 2.0 * math.pi * k / 3.0)
            roots.append(t - b / 3.0)
    
    return roots


def _poly_root_quartic(a, b, c, d, e):
    """
    Find real roots of ax^4 + bx^3 + cx^2 + dx + e = 0.
    Uses numerical approach via deflation.
    """
    if abs(a) < 1e-15:
        return _poly_root_cubic(b, c, d, e)
    
    # Normalize
    b = b / a
    c = c / a
    d = d / a
    e = e / a
    
    # Use companion matrix eigenvalue approach via numpy if available,
    # otherwise use a simple iterative method
    try:
        import numpy as np
        # Companion matrix
        comp = np.array([
            [0, 0, 0, -e],
            [1, 0, 0, -d],
            [0, 1, 0, -c],
            [0, 0, 1, -b]
        ])
        eigenvalues = np.linalg.eigvals(comp)
        real_roots = []
        for ev in eigenvalues:
            if abs(ev.imag) < 1e-10:
                real_roots.append(float(ev.real))
        return real_roots
    except:
        # Fallback: use Newton's method to find roots
        roots = []
        # Try multiple starting points
        for start in [-10, -1, 0, 1, 10]:
            x = float(start)
            for _ in range(50):
                fx = a * x**4 + b * x**3 + c * x**2 + d * x + e
                fpx = 4*a*x**3 + 3*b*x**2 + 2*c*x + d
                if abs(fpx) < 1e-15:
                    break
                x_new = x - fx / fpx
                if abs(x_new - x) < 1e-12:
                    break
                x = x_new
            fx = a * x**4 + b * x**3 + c * x**2 + d * x + e
            if abs(fx) < 1e-8:
                # Check if this root is already found
                is_new = True
                for r in roots:
                    if abs(r - x) < 1e-6:
                        is_new = False
                        break
                if is_new:
                    roots.append(x)
        return roots


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using Newton-FD fast path,
    polynomial interpolation, and adaptive bracketing strategies.
    """
    if not _is_finite(x0):
        x0 = 0.0
    x = float(x0)

    # Track best solution found so far
    best_x = x
    best_f = f(x)
    call_count = 1

    def safe_eval(x_val):
        nonlocal call_count, best_x, best_f
        if not _is_finite(x_val):
            return None
        fx_val = f(x_val)
        call_count += 1
        if not _is_finite(fx_val):
            return None
        if abs(fx_val) < abs(best_f):
            best_f = fx_val
            best_x = x_val
        return fx_val

    # Initial evaluation
    fx = best_f
    if abs(fx) <= 1e-8:
        return best_x

    # FAST PATH: Try secant method first (1 call/iter after initial 2)
    # Get a second point
    h0 = 1e-4 * max(1.0, abs(x))
    x2 = x + h0
    fx2 = safe_eval(x2)
    
    if fx2 is not None:
        # Try secant method
        x1, f1 = x, fx
        x2s, f2s = x2, fx2
        for _ in range(20):
            if call_count >= 480 or _time_left() <= 0:
                break
            if abs(f2s - f1) < 1e-15:
                # Fallback to bisection-like
                xm = (x1 + x2s) / 2.0
            else:
                xm = x2s - f2s * (x2s - x1) / (f2s - f1)
            if not _is_finite(xm):
                break
            
            fm = safe_eval(xm)
            if fm is None:
                break
            
            if abs(fm) <= 1e-8:
                return xm
            
            x1, f1 = x2s, f2s
            x2s, f2s = xm, fm
        
        if abs(best_f) <= 1e-8:
            return best_x
    
    # If secant didn't work, try Newton-FD
    newton_result = _newton_fd(f, x, fx0=fx, max_iter=4, tol=1e-8)
    if newton_result is not None:
        nx, nfx, ncalls = newton_result
        # Update call count and best
        call_count += ncalls
        if abs(nfx) < abs(best_f):
            best_f = nfx
            best_x = nx
        if abs(nfx) <= 1e-8:
            return nx

    # If we get here, try a broader search with polynomial interpolation
    points = [(x, fx)]
    
    # Collect more points
    h0 = 1e-4 * max(1.0, abs(x))
    for i in range(3, 5):
        if call_count >= 480 or _time_left() <= 0:
            break
        step = h0 * (10 ** (i - 3))
        xi = x + step
        fxi = safe_eval(xi)
        if fxi is not None:
            points.append((xi, fxi))
        
        xi = x - step
        fxi = safe_eval(xi)
        if fxi is not None:
            points.append((xi, fxi))
    
    # Remove duplicates
    unique_points = []
    for px, fpx in points:
        is_dup = False
        for ux, fux in unique_points:
            if abs(px - ux) < 1e-12:
                is_dup = True
                break
        if not is_dup:
            unique_points.append((px, fpx))
    
    points = unique_points
    
    # Try cubic polynomial interpolation
    if len(points) >= 4:
        sorted_pts = sorted(points, key=lambda p: abs(p[0] - x0))
        selected = sorted_pts[:4]
        
        try:
            import numpy as np
            A = np.vander([p[0] for p in selected], 4, increasing=True)
            y = np.array([p[1] for p in selected])
            coeffs, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
            roots = _poly_root_cubic(coeffs[3], coeffs[2], coeffs[1], coeffs[0])
            
            for r in roots:
                if not _is_finite(r):
                    continue
                if call_count >= 480 or _time_left() <= 0:
                    break
                fr = safe_eval(r)
                if fr is None:
                    continue
                if abs(fr) <= 1e-8:
                    return r
        except:
            pass

    # FINAL POLISH: If we have a good candidate, try to refine it
    if abs(best_f) <= 1e-6:
        # Try Newton polish from best_x
        x = best_x
        for _ in range(2):
            if call_count >= 490 or _time_left() <= 0:
                break
            h = 1e-7 * max(1.0, abs(x))
            fx_val = safe_eval(x)
            if fx_val is None:
                break
            if abs(fx_val) <= 1e-8:
                return x
            fx_plus = safe_eval(x + h)
            if fx_plus is None:
                break
            fpx = (fx_plus - fx_val) / h
            if abs(fpx) < 1e-15:
                break
            x_new = x - fx_val / fpx
            if not _is_finite(x_new):
                break
            if abs(x_new - x) < 1e-12 * max(1.0, abs(x)):
                break
            x = x_new
        if call_count < 490 and _time_left() > 0:
            fx_final = safe_eval(x)
            if fx_final is not None and abs(fx_final) <= 1e-8:
                return x

    return best_x
