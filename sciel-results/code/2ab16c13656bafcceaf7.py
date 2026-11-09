"""Direct root finder for smooth polynomials using polynomial interpolation + safeguarded Newton."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 145.0  # leave a margin before 150s kill
_SAFETY = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _cubic_interpolation_root(x0, x1, x2, x3, f0, f1, f2, f3):
    """Find the root of the cubic interpolant through (x0,f0),(x1,f1),(x2,f2),(x3,f3).
    Returns the x value closest to the point with smallest |f|."""
    # Build Lagrange basis polynomials and find where sum = 0
    # Use Newton's divided differences to build the interpolating polynomial
    # P(x) = f[x0] + f[x0,x1]*(x-x0) + f[x0,x1,x2]*(x-x0)*(x-x1) + f[x0,x1,x2,x3]*(x-x0)*(x-x1)*(x-x2)
    
    # Divided differences
    if abs(x1 - x0) < 1e-30:
        return None
    d01 = (f1 - f0) / (x1 - x0)
    if abs(x2 - x0) < 1e-30 or abs(x2 - x1) < 1e-30:
        # Fall back to quadratic
        return _quadratic_interpolation_root(x0, x1, x2, f0, f1, f2)
    d02 = (f2 - f0) / (x2 - x0)
    d12 = (f2 - f1) / (x2 - x1)
    d012 = (d12 - d01) / (x2 - x0)
    
    if abs(x3 - x0) < 1e-30 or abs(x3 - x1) < 1e-30 or abs(x3 - x2) < 1e-30:
        # Fall back to quadratic
        return _quadratic_interpolation_root(x0, x1, x2, f0, f1, f2)
    d03 = (f3 - f0) / (x3 - x0)
    d13 = (f3 - f1) / (x3 - x1)
    d23 = (f3 - f2) / (x3 - x2)
    d013 = (d13 - d01) / (x3 - x0)
    d123 = (d23 - d12) / (x3 - x1)
    d0123 = (d123 - d012) / (x3 - x0)
    
    # Now P(x) = f0 + d01*(x-x0) + d012*(x-x0)*(x-x1) + d0123*(x-x0)*(x-x1)*(x-x2)
    # This is a cubic in x. Let's expand and find roots.
    # Let u = x - x0, then x = u + x0
    # x - x1 = u + (x0-x1) = u + a1 where a1 = x0-x1
    # x - x2 = u + (x0-x2) = u + a2 where a2 = x0-x2
    # P = f0 + d01*u + d012*u*(u+a1) + d0123*u*(u+a1)*(u+a2)
    #   = f0 + d01*u + d012*(u^2 + a1*u) + d0123*(u^2 + (a1+a2)*u + a1*a2) * (u + a2) ... wait
    # Let me just expand directly:
    # u*(u+a1) = u^2 + a1*u
    # u*(u+a1)*(u+a2) = u*(u^2 + (a1+a2)*u + a1*a2) = u^3 + (a1+a2)*u^2 + a1*a2*u
    
    a1 = x0 - x1
    a2 = x0 - x2
    
    # Coefficients of u^3, u^2, u, constant:
    c3 = d0123
    c2 = d012 * 1.0 + d0123 * (a1 + a2)
    c1 = d01 + d012 * a1 + d0123 * a1 * a2
    c0 = f0
    
    # Solve c3*u^3 + c2*u^2 + c1*u + c0 = 0
    # Use numpy for cubic root finding if available, otherwise fallback
    try:
        import numpy as np
        coeffs = [c3, c2, c1, c0]
        roots = np.roots(coeffs)
        # Filter real roots
        real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
        if not real_roots:
            return None
        # Pick the root that gives the smallest |P(x)| and is finite
        best_u = None
        best_val = float('inf')
        for u in real_roots:
            x_candidate = u + x0
            if not _is_finite(x_candidate):
                continue
            # Evaluate P at x_candidate
            val = c3*u**3 + c2*u**2 + c1*u + c0
            if abs(val) < best_val:
                best_val = abs(val)
                best_u = u
        if best_u is not None:
            return best_u + x0
        return None
    except ImportError:
        # Fallback: use quadratic if cubic fails
        return _quadratic_interpolation_root(x0, x1, x2, f0, f1, f2)


def _quadratic_interpolation_root(x0, x1, x2, f0, f1, f2):
    """Find the root of the quadratic interpolant through (x0,f0),(x1,f1),(x2,f2)."""
    if abs(x1 - x0) < 1e-30 or abs(x2 - x0) < 1e-30 or abs(x2 - x1) < 1e-30:
        return None
    
    # Divided differences
    d01 = (f1 - f0) / (x1 - x0)
    d02 = (f2 - f0) / (x2 - x0)
    d12 = (f2 - f1) / (x2 - x1)
    d012 = (d12 - d01) / (x2 - x0)
    
    # P(x) = f0 + d01*(x-x0) + d012*(x-x0)*(x-x1)
    # Let u = x - x0
    # x - x1 = u + (x0-x1) = u + a1
    a1 = x0 - x1
    # P = f0 + d01*u + d012*u*(u+a1) = f0 + d01*u + d012*(u^2 + a1*u)
    #   = d012*u^2 + (d01 + d012*a1)*u + f0
    
    c2 = d012
    c1 = d01 + d012 * a1
    c0 = f0
    
    # Solve c2*u^2 + c1*u + c0 = 0
    if abs(c2) < 1e-30:
        # Linear
        if abs(c1) < 1e-30:
            return None
        u = -c0 / c1
    else:
        disc = c1*c1 - 4*c2*c0
        if disc < 0:
            return None
        sqrt_disc = math.sqrt(disc)
        u1 = (-c1 + sqrt_disc) / (2*c2)
        u2 = (-c1 - sqrt_disc) / (2*c2)
        # Pick the one that gives smaller |P|
        val1 = c2*u1*u1 + c1*u1 + c0
        val2 = c2*u2*u2 + c1*u2 + c0
        if abs(val1) <= abs(val2):
            u = u1
        else:
            u = u2
    
    x_candidate = u + x0
    if not _is_finite(x_candidate):
        return None
    return x_candidate


def _inverse_quadratic_interpolation(x0, x1, x2, f0, f1, f2):
    """Inverse quadratic interpolation: fit x as a quadratic function of f, then find x where f=0.
    This is often more robust than direct interpolation."""
    # Fit x = a*f^2 + b*f + c through (f0,x0), (f1,x1), (f2,x2)
    # Using Lagrange interpolation in f
    if abs(f1 - f0) < 1e-30 or abs(f2 - f0) < 1e-30 or abs(f2 - f1) < 1e-30:
        return None
    
    # Lagrange basis for f=0
    L0 = (0 - f1)*(0 - f2) / ((f0 - f1)*(f0 - f2))
    L1 = (0 - f0)*(0 - f2) / ((f1 - f0)*(f1 - f2))
    L2 = (0 - f0)*(0 - f1) / ((f2 - f0)*(f2 - f1))
    
    x_candidate = L0*x0 + L1*x1 + L2*x2
    if not _is_finite(x_candidate):
        return None
    return x_candidate


def find_root(f, x0):
    tol = 1e-8
    max_calls = 480
    calls = 0

    def ceval(x):
        """Counted, safe evaluation."""
        nonlocal calls
        if calls >= max_calls:
            return None, False
        calls += 1
        try:
            val = f(x)
        except Exception:
            return None, False
        if not _is_finite(val):
            return None, False
        return val, True

    # --- Phase 0: initial evaluation at x0 ---
    x = float(x0)
    fx, ok = ceval(x)
    if not ok:
        # probe a small grid around x0
        for dx in (1e-2, -1e-2, 1e-1, -1e-1, 1.0, -1.0, 10.0, -10.0):
            xt = x + dx
            fx, ok = ceval(xt)
            if ok:
                x = xt
                break
        if not ok:
            return x

    if abs(fx) <= tol:
        return x

    # --- Phase 1: Strategic sampling to build interpolation points ---
    # Sample 3 more points at geometrically spaced locations
    h0 = 1e-3 * max(1.0, abs(x))
    xs = [x]
    fs = [fx]
    
    # Try a few offsets to get diverse points
    offsets = [h0, -h0, 2*h0, -2*h0, 0.5*h0, -0.5*h0, 4*h0, -4*h0]
    for dx in offsets:
        if calls >= max_calls or len(xs) >= 5:
            break
        xt = x + dx
        ft, okt = ceval(xt)
        if okt:
            # Check if this point is sufficiently different from existing ones
            if all(abs(xt - xj) > 1e-12 * max(1.0, abs(xt), abs(xj)) for xj in xs):
                xs.append(xt)
                fs.append(ft)
                if abs(ft) <= tol:
                    return xt
    
    if len(xs) < 2:
        return x if abs(fx) <= tol else xs[0]

    best_x = xs[0]
    best_f = abs(fs[0])
    for xi, fi in zip(xs[1:], fs[1:]):
        if abs(fi) < best_f:
            best_f = abs(fi)
            best_x = xi

    if best_f <= tol:
        return best_x

    # --- Phase 2: Iterative polynomial interpolation refinement ---
    # Maintain a list of points, use interpolation to predict next candidate
    points = [(xs[i], fs[i]) for i in range(len(xs))]
    
    for iteration in range(30):
        if calls >= max_calls or _time_left() < 0:
            break
        
        # Choose interpolation method based on number of points
        n_pts = len(points)
        x_candidate = None
        
        if n_pts >= 4:
            # Try cubic interpolation
            x0i, f0i = points[0]
            x1i, f1i = points[1]
            x2i, f2i = points[2]
            x3i, f3i = points[3]
            x_candidate = _cubic_interpolation_root(x0i, x1i, x2i, x3i, f0i, f1i, f2i, f3i)
            
            if x_candidate is None:
                # Try inverse quadratic on the 3 points with smallest |f|
                sorted_pts = sorted(points, key=lambda p: abs(p[1]))
                x0i, f0i = sorted_pts[0]
                x1i, f1i = sorted_pts[1]
                x2i, f2i = sorted_pts[2]
                x_candidate = _inverse_quadratic_interpolation(x0i, x1i, x2i, f0i, f1i, f2i)
            
            if x_candidate is None:
                # Fall back to secant on the two best points
                sorted_pts = sorted(points, key=lambda p: abs(p[1]))
                x0i, f0i = sorted_pts[0]
                x1i, f1i = sorted_pts[1]
                if abs(f1i - f0i) > 1e-18:
                    x_candidate = x1i - f1i * (x1i - x0i) / (f1i - f0i)
        
        elif n_pts == 3:
            # Try quadratic interpolation
            x0i, f0i = points[0]
            x1i, f1i = points[1]
            x2i, f2i = points[2]
            x_candidate = _quadratic_interpolation_root(x0i, x1i, x2i, f0i, f1i, f2i)
            
            if x_candidate is None:
                # Try inverse quadratic
                x_candidate = _inverse_quadratic_interpolation(x0i, x1i, x2i, f0i, f1i, f2i)
            
            if x_candidate is None:
                # Secant on best two
                sorted_pts = sorted(points, key=lambda p: abs(p[1]))
                x0i, f0i = sorted_pts[0]
                x1i, f1i = sorted_pts[1]
                if abs(f1i - f0i) > 1e-18:
                    x_candidate = x1i - f1i * (x1i - x0i) / (f1i - f0i)
        
        elif n_pts == 2:
            # Secant
            x0i, f0i = points[0]
            x1i, f1i = points[1]
            if abs(f1i - f0i) > 1e-18:
                x_candidate = x1i - f1i * (x1i - x0i) / (f1i - f0i)
        
        if x_candidate is None or not _is_finite(x_candidate):
            # Try a small perturbation
            x_candidate = best_x + 1e-6 * max(1.0, abs(best_x))
        
        # Evaluate the candidate
        f_candidate, ok = ceval(x_candidate)
        if not ok:
            break
        
        if abs(f_candidate) <= tol:
            return x_candidate
        
        if abs(f_candidate) < best_f:
            best_f = abs(f_candidate)
            best_x = x_candidate
        
        # Update points: keep the best 5 points (by |f|) to maintain diversity
        points.append((x_candidate, f_candidate))
        if len(points) > 5:
            # Remove the worst point (largest |f|), but keep some diversity
            points.sort(key=lambda p: abs(p[1]))
            points = points[:5]
        
        # Check for convergence
        if best_f <= tol:
            return best_x

    if best_f <= tol:
        return best_x

    # --- Phase 3: Safeguarded Newton with finite-difference derivative ---
    xw = best_x
    for _ in range(10):
        if calls >= max_calls or _time_left() < 0:
            break
        f_w, ok = ceval(xw)
        if not ok:
            break
        if abs(f_w) <= tol:
            return xw
        if abs(f_w) < best_f:
            best_f = abs(f_w)
            best_x = xw

        # Compute derivative via central diff: 2 calls
        h = 1e-6 * max(1.0, abs(xw))
        fp, ok1 = ceval(xw + h)
        fm, ok2 = ceval(xw - h)
        if not (ok1 and ok2):
            if ok1:
                d = (fp - f_w) / h
            elif ok2:
                d = (f_w - fm) / h
            else:
                break
        else:
            d = (fp - fm) / (2 * h)

        if abs(d) < 1e-18:
            break

        x_newton = xw - f_w / d
        if not _is_finite(x_newton):
            break

        step = x_newton - xw
        max_step = 100.0 * max(1.0, abs(xw))
        if abs(step) > max_step:
            x_newton = xw + max_step * math.copysign(1.0, step)

        f_newton, ok = ceval(x_newton)
        if not ok:
            break
        if abs(f_newton) <= tol:
            return x_newton
        if abs(f_newton) < best_f:
            best_f = abs(f_newton)
            best_x = x_newton
        xw = x_newton

    if best_f <= tol:
        return best_x

    # --- Phase 4: Bisection fallback ---
    x_lo, x_hi = best_x, best_x
    f_lo, ok = ceval(x_lo)
    if not ok:
        return best_x
    f_hi = f_lo
    step = 0.1 * max(1.0, abs(best_x))
    bracketed = False
    for _ in range(20):
        if calls >= max_calls or _time_left() < 0:
            break
        x_try = x_lo - step
        f_try, ok = ceval(x_try)
        if ok and f_lo * f_try < 0:
            x_hi = x_lo
            f_hi = f_lo
            x_lo = x_try
            f_lo = f_try
            bracketed = True
            break
        x_try = x_hi + step
        f_try, ok = ceval(x_try)
        if ok and f_lo * f_try < 0:
            x_hi = x_try
            f_hi = f_try
            bracketed = True
            break
        step *= 2.0

    if bracketed:
        for _ in range(50):
            if calls >= max_calls or _time_left() < 0:
                break
            x_mid = 0.5 * (x_lo + x_hi)
            f_mid, ok = ceval(x_mid)
            if not ok:
                break
            if abs(f_mid) <= tol:
                return x_mid
            if abs(f_mid) < best_f:
                best_f = abs(f_mid)
                best_x = x_mid
            if f_lo * f_mid <= 0:
                x_hi = x_mid
                f_hi = f_mid
            else:
                x_lo = x_mid
                f_lo = f_mid

    return best_x
