"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Polynomial-interpolation root finder for smooth polynomial equations.
    Uses cubic interpolation on 4 points when possible, falling back to
    secant/bisection. Minimizes function calls by leveraging the fact that
    the hidden functions are smooth polynomials.
    """
    t_start = time.time()
    DEADLINE = t_start + 115.0  # safety margin before 150s kill

    def time_ok():
        return time.time() < DEADLINE

    x0 = float(x0)
    best_x = x0
    best_f = f(x0)
    best_abs = abs(best_f)

    def update_best(xc, fc):
        nonlocal best_x, best_f, best_abs
        if math.isfinite(xc) and math.isfinite(fc):
            af = abs(fc)
            if af < best_abs:
                best_x, best_f, best_abs = xc, fc, af

    if best_abs <= 1e-8:
        return best_x

    # Store evaluated points: list of (x, f(x))
    points = [(x0, best_f)]

    def add_point(xc, fc):
        points.append((xc, fc))
        update_best(xc, fc)

    # --- Phase 1: Cubic interpolation with adaptive point selection ---
    # Start with x0, then add points nearby to build interpolation
    # For polynomials of degree <= 3, 4 points give exact root
    
    # Generate candidate points near x0
    scale = max(1.0, abs(x0))
    candidates = [
        x0 + 0.1 * scale,
        x0 - 0.1 * scale,
        x0 + 1.0 * scale,
        x0 - 1.0 * scale,
    ]
    
    for cx in candidates:
        if not time_ok():
            break
        if best_abs <= 1e-8:
            return best_x
        if not math.isfinite(cx):
            continue
        try:
            cf = f(cx)
        except Exception:
            continue
        if not math.isfinite(cf):
            continue
        add_point(cx, cf)
        if best_abs <= 1e-8:
            return best_x

    # Now we have up to 5 points. Use cubic interpolation if we have >= 4.
    # Select best 4 points (closest to x0 or with smallest |f|)
    def cubic_root_from_4(pts):
        """Find root of cubic interpolating 4 points. Returns root or None."""
        if len(pts) < 4:
            return None
        # Select 4 points: prefer those with smallest |f| and spread
        # Simple strategy: take first 4 if available
        xs = [p[0] for p in pts[:4]]
        fs = [p[1] for p in pts[:4]]
        
        # Lagrange interpolation to get cubic coefficients
        # P(x) = sum_i f(x_i) * L_i(x) where L_i are Lagrange basis polynomials
        # Then solve P(x) = 0
        
        # Build the cubic polynomial in standard form: a*x^3 + b*x^2 + c*x + d
        # Using Newton's divided differences for numerical stability
        n = 4
        # Divided differences table
        dd = [[0.0] * n for _ in range(n)]
        for i in range(n):
            dd[i][i] = fs[i]
        
        for j in range(1, n):
            for i in range(n - j):
                if abs(xs[i + j] - xs[i]) < 1e-15:
                    return None  # Degenerate
                dd[i][i + j] = (dd[i + 1][i + j] - dd[i][i + j - 1]) / (xs[i + j] - xs[i])
        
        # Newton form: P(x) = dd[0][0] + dd[0][1]*(x-x0) + dd[0][2]*(x-x0)(x-x1) + dd[0][3]*(x-x0)(x-x1)(x-x2)
        # Convert to standard polynomial coefficients
        # Start with constant
        coeffs = [dd[0][0]]  # d0
        for j in range(1, n):
            # Multiply current poly by (x - xs[j-1]) and add dd[0][j]
            new_coeffs = [0.0] * (len(coeffs) + 1)
            for k in range(len(coeffs)):
                new_coeffs[k] += coeffs[k] * (-xs[j-1])
                new_coeffs[k+1] += coeffs[k]
            new_coeffs[0] += dd[0][j]
            coeffs = new_coeffs
        
        # coeffs[0] + coeffs[1]*x + coeffs[2]*x^2 + coeffs[3]*x^3
        a = coeffs[3]
        b = coeffs[2]
        c = coeffs[1]
        d = coeffs[0]
        
        if abs(a) < 1e-15:
            # Quadratic
            if abs(b) < 1e-15:
                # Linear
                if abs(c) < 1e-15:
                    return None
                return -d / c
            disc = c * c - 4 * b * d
            if disc < 0:
                return None
            sq = math.sqrt(disc)
            r1 = (-c + sq) / (2 * b)
            r2 = (-c - sq) / (2 * b)
            # Return the one closer to x0
            if abs(r1 - x0) < abs(r2 - x0):
                return r1
            return r2
        
        # Cubic: a*x^3 + b*x^2 + c*x + d = 0
        # Use numpy if available, else analytical
        try:
            import numpy as np
            roots = np.roots([a, b, c, d])
            real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
            if not real_roots:
                return None
            # Return root closest to x0
            return min(real_roots, key=lambda r: abs(r - x0))
        except ImportError:
            # Fallback: use bisection on the cubic if we can bracket
            return None

    # Try cubic interpolation
    if len(points) >= 4:
        root = cubic_root_from_4(points)
        if root is not None and math.isfinite(root):
            try:
                fr = f(root)
            except Exception:
                fr = None
            if fr is not None and math.isfinite(fr):
                add_point(root, fr)
                if best_abs <= 1e-8:
                    return best_x
    
    # --- Phase 2: Secant method iterations with point reuse ---
    # Use the two best points (smallest |f|) as secant starting points
    if len(points) >= 2:
        # Sort by |f|
        sorted_pts = sorted(points, key=lambda p: abs(p[1]))
        x1, f1 = sorted_pts[0]
        x2, f2 = sorted_pts[1]
        
        for i in range(20):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(f1 - f2) < 1e-15:
                break
            # Secant step
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            if not math.isfinite(x_new):
                break
            # Check if we've already evaluated this point
            already = False
            for px, pf in points:
                if abs(px - x_new) < 1e-12 * max(1.0, abs(x_new)):
                    already = True
                    break
            if already:
                break
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            add_point(x_new, f_new)
            if best_abs <= 1e-8:
                return best_x
            # Update: keep the two most recent for secant
            if abs(f_new) < abs(f2):
                x2, f2 = x1, f1
                x1, f1 = x_new, f_new
            else:
                x2, f2 = x_new, f_new
    
    # --- Phase 3: Bracketing search if not converged ---
    if best_abs > 1e-8:
        found_bracket = False
        a = b = None
        fa = fb = None
        
        for scale in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e5]:
            if not time_ok():
                break
            half = scale * max(1.0, abs(x0))
            a = x0 - half
            b = x0 + half
            try:
                fa = f(a)
                fb = f(b)
            except Exception:
                continue
            if not (math.isfinite(fa) and math.isfinite(fb)):
                continue
            add_point(a, fa)
            add_point(b, fb)
            if best_abs <= 1e-8:
                return best_x
            if fa * fb < 0:
                found_bracket = True
                break
        
        if found_bracket and a is not None and b is not None:
            # Bisection
            ca, cb = a, b
            fc_a, fc_b = fa, fb
            for i in range(60):
                if not time_ok():
                    break
                mid = 0.5 * (ca + cb)
                try:
                    fm = f(mid)
                except Exception:
                    break
                if not math.isfinite(fm):
                    break
                add_point(mid, fm)
                if best_abs <= 1e-8:
                    return best_x
                if fm * fc_a < 0:
                    cb = mid
                    fc_b = fm
                else:
                    ca = mid
                    fc_a = fm
                if abs(cb - ca) < 1e-14 * max(1.0, abs(mid)):
                    break
    
    # --- Phase 4: Final polish with secant ---
    if best_abs > 1e-8 and len(points) >= 2:
        sorted_pts = sorted(points, key=lambda p: abs(p[1]))
        x1, f1 = sorted_pts[0]
        x2, f2 = sorted_pts[1]
        for i in range(10):
            if not time_ok():
                break
            if best_abs <= 1e-8:
                return best_x
            if abs(f1 - f2) < 1e-15:
                break
            x_new = x1 - f1 * (x1 - x2) / (f1 - f2)
            if not math.isfinite(x_new):
                break
            try:
                f_new = f(x_new)
            except Exception:
                break
            if not math.isfinite(f_new):
                break
            add_point(x_new, f_new)
            if abs(f_new) < abs(f2):
                x2, f2 = x1, f1
                x1, f1 = x_new, f_new
            else:
                x2, f2 = x_new, f_new

    return best_x
# EVOLVE-BLOCK-END
