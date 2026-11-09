"""Root finder using cubic interpolation with secant fallback."""
import time
import math

_DEADLINE_START = time.time()
_WALL_LIMIT = 115.0
_SAFETY_MARGIN = 5.0


def _time_left():
    return _WALL_LIMIT - _SAFETY_MARGIN - (time.time() - _DEADLINE_START)


def _is_finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _cubic_root(x0, f0, x1, f1, x2, f2, x3, f3):
    """
    Find root of cubic polynomial interpolating 4 points.
    Returns the root closest to x1 (or any finite root if all similar).
    Uses Lagrange interpolation coefficients.
    """
    # Build cubic polynomial P(x) = a*x^3 + b*x^2 + c*x + d
    # using Lagrange basis polynomials
    # P(x) = sum_i f_i * L_i(x) where L_i(x) = prod_{j!=i} (x - x_j)/(x_i - x_j)
    
    # For numerical stability, use divided differences (Newton form)
    # P(x) = f[x0] + f[x0,x1]*(x-x0) + f[x0,x1,x2]*(x-x0)(x-x1) + f[x0,x1,x2,x3]*(x-x0)(x-x1)(x-x2)
    
    def div_diff(xs, fs):
        n = len(xs)
        # Compute divided differences
        dd = [[0.0]*n for _ in range(n)]
        for i in range(n):
            dd[i][0] = fs[i]
        for j in range(1, n):
            for i in range(n - j):
                if abs(xs[i+j] - xs[i]) < 1e-15:
                    dd[i][j] = 0.0
                else:
                    dd[i][j] = (dd[i+1][j-1] - dd[i][j-1]) / (xs[i+j] - xs[i])
        return dd
    
    xs = [x0, x1, x2, x3]
    fs = [f0, f1, f2, f3]
    
    dd = div_diff(xs, fs)
    
    # Newton form: P(x) = dd[0][0] + dd[0][1]*(x-x0) + dd[0][2]*(x-x0)(x-x1) + dd[0][3]*(x-x0)(x-x1)(x-x2)
    a = dd[0][3]  # coefficient of (x-x0)(x-x1)(x-x2)
    b = dd[0][2]
    c = dd[0][1]
    d = dd[0][0]
    
    # Expand to standard form: a*x^3 + b2*x^2 + c2*x + d2
    # (x-x0)(x-x1)(x-x2) = x^3 - (x0+x1+x2)*x^2 + (x0*x1+x0*x2+x1*x2)*x - x0*x1*x2
    s1 = x0 + x1 + x2
    s2 = x0*x1 + x0*x2 + x1*x2
    s3 = x0*x1*x2
    
    # (x-x0)(x-x1) = x^2 - (x0+x1)*x + x0*x1
    t1 = x0 + x1
    t2 = x0*x1
    
    # P(x) = a*(x^3 - s1*x^2 + s2*x - s3) + b*(x^2 - t1*x + t2) + c*(x - x0) + d
    A = a
    B = -a*s1 + b
    C = a*s2 - b*t1 + c
    D = -a*s3 + b*t2 - c*x0 + d
    
    # Solve A*x^3 + B*x^2 + C*x + D = 0
    # If A is very small, it's quadratic or linear
    if abs(A) < 1e-15:
        if abs(B) < 1e-15:
            if abs(C) < 1e-15:
                return None
            return -D / C
        # Quadratic: B*x^2 + C*x + D = 0
        disc = C*C - 4*B*D
        if disc < -1e-15:
            return None
        disc = max(0.0, disc)
        sq = math.sqrt(disc)
        r1 = (-C + sq) / (2*B)
        r2 = (-C - sq) / (2*B)
        # Return root closest to x1
        if abs(r1 - x1) <= abs(r2 - x1):
            return r1
        return r2
    
    # Cubic: A*x^3 + B*x^2 + C*x + D = 0
    # Normalize: x^3 + p*x^2 + q*x + r = 0
    p = B / A
    q = C / A
    r = D / A
    
    # Depressed cubic: x = t - p/3
    # t^3 + pt' + q' = 0 where p' = q - p^2/3, q' = 2p^3/27 - p*q/3 + r
    p_prime = q - p*p/3.0
    q_prime = 2.0*p**3/27.0 - p*q/3.0 + r
    
    # Discriminant of depressed cubic
    disc = (q_prime/2.0)**2 + (p_prime/3.0)**3
    
    roots = []
    if disc > 1e-15:
        # One real root
        sq_disc = math.sqrt(disc)
        u = -q_prime/2.0 + sq_disc
        v = -q_prime/2.0 - sq_disc
        # Cube roots (handle negative)
        def cbrt(x):
            if x >= 0:
                return x**(1.0/3.0)
            else:
                return -((-x)**(1.0/3.0))
        t = cbrt(u) + cbrt(v)
        roots.append(t - p/3.0)
    elif abs(disc) <= 1e-15:
        # Multiple roots
        if abs(p_prime) < 1e-15:
            # Triple root
            roots.append(-p/3.0)
        else:
            # Double root and single root
            t1 = 3.0*q_prime/p_prime
            t2 = -3.0*q_prime/(2.0*p_prime)
            roots.append(t1 - p/3.0)
            roots.append(t2 - p/3.0)
    else:
        # Three distinct real roots
        # trigonometric method
        rho = math.sqrt(-p_prime**3/27.0)
        theta = math.acos(max(-1.0, min(1.0, -q_prime/(2.0*rho))))
        mag = 2.0 * (-p_prime/3.0)**0.5
        for k in range(3):
            t = mag * math.cos(theta/3.0 - 2.0*math.pi*k/3.0)
            roots.append(t - p/3.0)
    
    if not roots:
        return None
    
    # Return root closest to x1
    best = min(roots, key=lambda r: abs(r - x1))
    return best if _is_finite(best) else None


def find_root(f, x0):
    """
    Find x such that |f(x)| <= 1e-8 using cubic interpolation with secant fallback.
    """
    tol = 1e-12
    max_iter = 80
    
    x0_val = float(x0)
    try:
        f0 = f(x0_val)
    except Exception:
        return x0_val
    
    if not _is_finite(f0):
        return x0_val
    
    best_x = x0_val
    best_f = abs(f0)
    
    if best_f <= tol:
        return x0_val
    
    # Generate initial points for cubic interpolation
    # Use geometrically spaced points to bracket the root
    scale = max(1.0, abs(x0_val))
    h1 = 1e-4 * scale
    h2 = 1e-3 * scale
    h3 = 1e-2 * scale
    
    points = [(x0_val, f0)]
    
    for h in [h1, h2, h3]:
        if _time_left() <= 0:
            break
        x_try = x0_val + h
        try:
            f_try = f(x_try)
        except Exception:
            break
        if not _is_finite(f_try):
            continue
        abs_f_try = abs(f_try)
        if abs_f_try < best_f:
            best_x = x_try
            best_f = abs_f_try
        points.append((x_try, f_try))
        if best_f <= tol:
            return x_try
    
    # Also try negative direction
    for h in [-h1, -h2, -h3]:
        if len(points) >= 4:
            break
        if _time_left() <= 0:
            break
        x_try = x0_val + h
        try:
            f_try = f(x_try)
        except Exception:
            break
        if not _is_finite(f_try):
            continue
        abs_f_try = abs(f_try)
        if abs_f_try < best_f:
            best_x = x_try
            best_f = abs_f_try
        points.append((x_try, f_try))
        if best_f <= tol:
            return x_try
    
    if len(points) < 4:
        # Fall back to secant with whatever points we have
        if len(points) >= 2:
            return _secant_fallback(f, points, best_x, best_f, tol, max_iter)
        return best_x
    
    # Use first 4 points for cubic interpolation
    (xa, fa), (xb, fb), (xc, fc), (xd, fd) = points[0], points[1], points[2], points[3]
    
    for _ in range(max_iter):
        if _time_left() <= 0:
            break
        
        root = _cubic_root(xa, fa, xb, fb, xc, fc, xd, fd)
        
        if root is None or not _is_finite(root):
            # Fall back to secant
            return _secant_fallback(f, points, best_x, best_f, tol, max_iter)
        
        try:
            f_root = f(root)
        except Exception:
            break
        
        if not _is_finite(f_root):
            # Shift window
            points = points[1:] + [(root, f_root)]
            continue
        
        abs_f_root = abs(f_root)
        if abs_f_root < best_f:
            best_x = root
            best_f = abs_f_root
        
        if abs_f_root <= tol:
            return root
        
        # Shift window: drop oldest point, add new one
        points = points[1:] + [(root, f_root)]
        (xa, fa), (xb, fb), (xc, fc), (xd, fd) = points[0], points[1], points[2], points[3]
        
        # Check for convergence stall
        if len(points) > 2 and abs(points[-1][0] - points[-2][0]) < 1e-14:
            break
    
    return best_x


def _secant_fallback(f, points, best_x, best_f, tol, max_iter):
    """Secant method fallback using the last two points."""
    if len(points) < 2:
        return best_x
    
    x_prev, f_prev = points[-2]
    x_curr, f_curr = points[-1]
    
    for _ in range(max_iter):
        if _time_left() <= 0:
            break
        
        dx = x_curr - x_prev
        if dx == 0:
            break
        
        m = (f_curr - f_prev) / dx
        if m == 0 or not _is_finite(m):
            break
        
        x_next = x_curr - f_curr / m
        if not _is_finite(x_next):
            break
        
        try:
            f_next = f(x_next)
        except Exception:
            break
        
        if not _is_finite(f_next):
            break
        
        abs_f_next = abs(f_next)
        if abs_f_next < best_f:
            best_x = x_next
            best_f = abs_f_next
        
        if abs_f_next <= tol:
            return x_next
        
        x_prev, f_prev = x_curr, f_curr
        x_curr, f_curr = x_next, f_next
    
    return best_x
