"""Scalar root finder for 3 fixed smooth polynomial equations."""
import math
import time

# EVOLVE-BLOCK-START
def find_root(f, x0):
    """
    Find a root of f starting near x0, returning x with |f(x)| <= 1e-8.
    Uses a hybrid approach: polynomial fitting, Brent's method, and Newton refinement.
    """
    DEADLINE = 148.0
    start_time = time.time()

    def check_deadline():
        return (time.time() - start_time) < DEADLINE

    def is_finite(x):
        return math.isfinite(x)

    def safe_f(x):
        try:
            val = f(x)
            if not is_finite(val):
                return None
            return float(val)
        except:
            return None

    best_x = None
    best_fx = float('inf')

    def update_best(x, fx):
        nonlocal best_x, best_fx
        if is_finite(x) and is_finite(fx) and abs(fx) < abs(best_fx):
            best_x = x
            best_fx = fx
        return best_x is not None and abs(best_fx) <= 1e-8

    x0 = float(x0)
    fx0 = safe_f(x0)
    if fx0 is not None:
        if update_best(x0, fx0):
            return x0

    # Strategy 1: Polynomial fitting with multiple point sets
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        scale = max(1.0, abs(x0))
        
        # Try different point configurations
        point_sets = [
            [x0 - 2*scale, x0 - scale, x0, x0 + scale, x0 + 2*scale],
            [x0 - 4*scale, x0 - 2*scale, x0, x0 + 2*scale, x0 + 4*scale],
            [x0 - 8*scale, x0 - 4*scale, x0, x0 + 4*scale, x0 + 8*scale],
            [x0 - 3*scale, x0 - scale, x0 + scale, x0 + 2*scale, x0 + 3*scale],
            [x0 - 1*scale, x0 - 0.5*scale, x0, x0 + 0.5*scale, x0 + scale],
        ]
        
        for pts in point_sets:
            if not check_deadline():
                break
            
            vals = []
            valid = True
            for p in pts:
                v = safe_f(p)
                if v is None:
                    valid = False
                    break
                vals.append(v)
            
            if not valid:
                continue
            
            # Try degrees 2, 3, 4
            for deg in [2, 3, 4]:
                if deg + 1 > len(pts):
                    continue
                fit_pts = pts[:deg+1]
                fit_vals = vals[:deg+1]
                
                try:
                    # Build Vandermonde matrix
                    n = deg + 1
                    A = [[p**j for j in range(n)] for p in fit_pts]
                    b = fit_vals
                    
                    # Solve using Gaussian elimination
                    # Augmented matrix
                    aug = [A[i] + [b[i]] for i in range(n)]
                    
                    for i in range(n):
                        # Find pivot
                        max_row = i
                        for r in range(i+1, n):
                            if abs(aug[r][i]) > abs(aug[max_row][i]):
                                max_row = r
                        if abs(aug[max_row][i]) < 1e-15:
                            continue
                        aug[i], aug[max_row] = aug[max_row], aug[i]
                        
                        # Eliminate
                        for r in range(i+1, n):
                            factor = aug[r][i] / aug[i][i]
                            for c in range(i, n+1):
                                aug[r][c] -= factor * aug[i][c]
                    
                    # Back substitution
                    coeffs = [0.0] * n
                    for i in range(n-1, -1, -1):
                        s = aug[i][n]
                        for j in range(i+1, n):
                            s -= aug[i][j] * coeffs[j]
                        coeffs[i] = s / aug[i][i]
                    
                    # Find roots of polynomial
                    roots = _find_poly_roots(coeffs)
                    
                    for r in roots:
                        if not is_finite(r):
                            continue
                        fr = safe_f(r)
                        if fr is None:
                            continue
                        if abs(fr) <= 1e-8:
                            return r
                        update_best(r, fr)
                        if abs(fr) <= 1e-8:
                            return r
                except:
                    continue
        
        if best_x is not None and abs(best_fx) > 1e-8 and check_deadline():
            # Refine best candidate with secant
            x1 = best_x
            fx1 = best_fx
            x2 = x1 + 0.1 * max(1.0, abs(x1))
            fx2 = safe_f(x2)
            
            if fx2 is not None:
                update_best(x2, fx2)
                if abs(fx2) <= 1e-8:
                    return x2
                
                for _ in range(100):
                    if not check_deadline():
                        break
                    if abs(fx2) <= 1e-8:
                        return x2
                    if abs(fx1) <= 1e-8:
                        return x1
                    
                    denom = fx2 - fx1
                    if denom == 0 or not is_finite(denom):
                        break
                    x_new = x2 - fx2 * (x2 - x1) / denom
                    if not is_finite(x_new):
                        break
                    if abs(x_new - x1) < 1e-12:
                        x_new = x1 + 1e-6
                    if abs(x_new - x2) < 1e-12:
                        x_new = x2 + 1e-6
                    
                    x1, fx1 = x2, fx2
                    x2 = x_new
                    fx2 = safe_f(x2)
                    if fx2 is None:
                        break

    # Strategy 2: Brent's method with bracketing
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        # Find a bracket
        a, b, fa, fb = None, None, None, None
        scale = 0.1 * max(1.0, abs(x0))
        
        for _ in range(60):
            if not check_deadline():
                break
            a_try = x0 - scale
            b_try = x0 + scale
            
            fa_try = safe_f(a_try)
            fb_try = safe_f(b_try)
            
            if fa_try is not None and fb_try is not None:
                if fa_try * fb_try <= 0:
                    a, b, fa, fb = a_try, b_try, fa_try, fb_try
                    break
                if abs(fa_try) <= 1e-8:
                    return a_try
                if abs(fb_try) <= 1e-8:
                    return b_try
            
            scale *= 2
            if scale > 1e10:
                break
        
        if a is not None and b is not None:
            # Brent's method
            x = b
            fx = fb
            for _ in range(200):
                if not check_deadline():
                    break
                if abs(fx) <= 1e-8:
                    return x
                update_best(x, fx)
                
                # Check if we've converged
                if abs(b - a) < 1e-12:
                    break
                
                # Try bisection
                m = (a + b) / 2.0
                fm = safe_f(m)
                if fm is None:
                    break
                if abs(fm) <= 1e-8:
                    return m
                update_best(m, fm)
                
                # Choose between bisection and inverse quadratic interpolation
                if fa * fm <= 0:
                    b = m
                    fb = fm
                else:
                    a = m
                    fa = fm
                
                # Use secant step if possible
                if fa * fb > 0:
                    # No sign change, use bisection
                    x = (a + b) / 2.0
                else:
                    # Try secant
                    denom = fb - fa
                    if abs(denom) > 1e-15:
                        x = b - fb * (b - a) / denom
                        if x < min(a, b) or x > max(a, b):
                            x = (a + b) / 2.0
                    
                    fx = safe_f(x)
                    if fx is None:
                        x = (a + b) / 2.0
                        fx = safe_f(x)
                        if fx is None:
                            break
                    update_best(x, fx)

    # Strategy 3: Newton's method with finite-difference derivative
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        x = x0
        for _ in range(150):
            if not check_deadline():
                break
            fx = safe_f(x)
            if fx is None:
                break
            if abs(fx) <= 1e-8:
                return x
            update_best(x, fx)
            
            h = 1e-7 * max(1.0, abs(x))
            fx_plus = safe_f(x + h)
            if fx_plus is None:
                break
            d = (fx_plus - fx) / h
            if d == 0 or not is_finite(d):
                break
            x_new = x - fx / d
            if not is_finite(x_new):
                break
            if abs(x_new - x) > 10 * max(1.0, abs(x)):
                x_new = x + 0.5 * (x_new - x)
            x = x_new

    # Strategy 4: Global search with multiple starting points
    if check_deadline() and best_x is not None and abs(best_fx) > 1e-8:
        offsets = [0.01, -0.01, 0.1, -0.1, 1.0, -1.0, 10.0, -10.0,
                   0.001, -0.001, 100.0, -100.0, 0.5, -0.5, 5.0, -5.0,
                   50.0, -50.0, 0.05, -0.05]
        for offset in offsets:
            if not check_deadline():
                break
            x_start = x0 + offset * max(1.0, abs(x0))
            fx_start = safe_f(x_start)
            if fx_start is None:
                continue
            if abs(fx_start) <= 1e-8:
                return x_start
            update_best(x_start, fx_start)
            
            # Quick secant refinement
            x1 = x_start
            fx1 = fx_start
            x2 = x_start + 0.1 * max(1.0, abs(x_start))
            fx2 = safe_f(x2)
            if fx2 is None:
                continue
            update_best(x2, fx2)
            if abs(fx2) <= 1e-8:
                return x2
            
            for _ in range(50):
                if not check_deadline():
                    break
                if abs(fx2) <= 1e-8:
                    return x2
                if abs(fx1) <= 1e-8:
                    return x1
                
                denom = fx2 - fx1
                if denom == 0 or not is_finite(denom):
                    break
                x_new = x2 - fx2 * (x2 - x1) / denom
                if not is_finite(x_new):
                    break
                if abs(x_new - x1) < 1e-12:
                    x_new = x1 + 1e-6
                if abs(x_new - x2) < 1e-12:
                    x_new = x2 + 1e-6
                
                x1, fx1 = x2, fx2
                x2 = x_new
                fx2 = safe_f(x2)
                if fx2 is None:
                    break

    if best_x is not None:
        return best_x
    return x0


def _find_poly_roots(coeffs):
    """Find real roots of a polynomial given coefficients from highest to lowest degree."""
    # Remove leading zeros
    while len(coeffs) > 1 and abs(coeffs[0]) < 1e-15:
        coeffs = coeffs[1:]
    
    if len(coeffs) == 1:
        return []
    if len(coeffs) == 2:
        # Linear: ax + b = 0
        if abs(coeffs[0]) > 1e-15:
            return [-coeffs[1] / coeffs[0]]
        return []
    
    # Use companion matrix eigenvalue approach
    n = len(coeffs) - 1
    if n < 1:
        return []
    
    # Build companion matrix
    # For polynomial a_n x^n + a_{n-1} x^{n-1} + ... + a_0
    # Companion matrix has -a_i/a_n on the superdiagonal and last column -a_{n-1}/a_n, ..., -a_0/a_n
    lead = coeffs[0]
    if abs(lead) < 1e-15:
        return []
    
    # Normalize
    c = [ci / lead for ci in coeffs]
    
    # Companion matrix
    comp = [[0.0] * n for _ in range(n)]
    for i in range(n - 1):
        comp[i][i + 1] = -c[i + 1]
    for i in range(n):
        comp[n - 1][i] = -c[n - i]
    
    # Find eigenvalues using QR algorithm
    eigvals = _qr_eigenvalues(comp)
    
    real_roots = []
    for ev in eigvals:
        if abs(ev[1]) < 1e-10:
            real_roots.append(ev[0])
    
    return real_roots


def _qr_eigenvalues(A):
    """Find eigenvalues of a real matrix using QR iteration."""
    n = len(A)
    if n == 0:
        return []
    if n == 1:
        return [(A[0][0], 0.0)]
    if n == 2:
        # 2x2 eigenvalue formula
        a, b = A[0][0], A[0][1]
        c, d = A[1][0], A[1][1]
        trace = a + d
        det = a * d - b * c
        disc = trace * trace - 4 * det
        if disc >= 0:
            sqrt_disc = math.sqrt(disc)
            l1 = (trace + sqrt_disc) / 2
            l2 = (trace - sqrt_disc) / 2
            return [(l1, 0.0), (l2, 0.0)]
        else:
            real = trace / 2
            imag = math.sqrt(-disc) / 2
            return [(real, imag), (real, -imag)]
    
    # QR iteration with shifts
    M = [row[:] for row in A]
    max_iter = 1000
    
    for _ in range(max_iter):
        # Check for convergence
        off_diag = 0.0
        for i in range(n - 1):
            off_diag += abs(M[i + 1][i]) ** 2
        if off_diag < 1e-20:
            break
        
        # QR decomposition (simplified: use Householder or Givens)
        Q, R = _qr_decompose(M)
        M = _matmul(R, Q)
    
    # Eigenvalues are on the diagonal
    eigvals = []
    for i in range(n):
        eigvals.append((M[i][i], 0.0))
    
    return eigvals


def _qr_decompose(A):
    """QR decomposition using Householder reflections."""
    n = len(A)
    R = [row[:] for row in A]
    Q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    for k in range(n - 1):
        # Find Householder vector
        x = [R[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < 1e-15:
            continue
        
        # Choose sign to avoid cancellation
        if x[0] >= 0:
            v = [x[0] + norm_x] + x[1:]
        else:
            v = [x[0] - norm_x] + x[1:]
        
        v_norm = math.sqrt(sum(vi * vi for vi in v))
        if v_norm < 1e-15:
            continue
        v = [vi / v_norm for vi in v]
        
        # Apply Householder reflection to R
        for j in range(k, n):
            dot = sum(v[i] * R[k + i][j] for i in range(n - k))
            for i in range(k, n):
                R[i][j] -= 2 * v[i - k] * dot
        
        # Apply to Q
        for i in range(n):
            dot = sum(v[j] * Q[i][k + j] for j in range(n - k))
            for j in range(k, n):
                Q[i][j] -= 2 * v[j - k] * dot
    
    return Q, R


def _matmul(A, B):
    """Matrix multiplication."""
    n = len(A)
    m = len(B[0])
    p = len(B)
    C = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for k in range(p):
                s += A[i][k] * B[k][j]
            C[i][j] = s
    return C
# EVOLVE-BLOCK-END
