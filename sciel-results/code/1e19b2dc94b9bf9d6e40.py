"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, laguerre
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(laguerre(d, x, alpha))
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols[k].diff(x).subs(x, 0)
    b[1, 0] = 1
    
    # Rows 2..25
    for i in range(12):
        zi = z_rationals[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].subs(x, zi)
        b[row_idx, 0] = 0
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = laguerre_pols[k].diff(x).subs(x, zi)
        b[row_idx, 0] = 0
    
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, _ = g_poly.div(denom_poly)
    except Exception:
        return None
    
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    eps = Rational(1, 10)**198
    
    max_r = 0
    for r in real_roots_list:
        try:
            q_at_minus = q_expr.subs(x, r - eps)
            q_at_plus = q_expr.subs(x, r + eps)
            if q_at_minus * q_at_plus < 0:
                if r > max_r:
                    max_r = r
        except Exception:
            continue
    
    B = max(0, max_r)
    score = B / (2 * sympy_pi)
    return float(score)

def _get_laguerre_coeffs(n, alpha):
    """Return coefficients of L_n^(alpha)(x) in ascending power of x."""
    # L_n^(alpha)(x) = sum_{k=0}^n binom(n+alpha, n-k) (-1)^k x^k / k!
    coeffs = np.zeros(n + 1)
    for k in range(n + 1):
        # Use log-gamma for stability if needed, but n is small (<=50)
        # binom(n+alpha, n-k) = Gamma(n+alpha+1) / (Gamma(n-k+1) * Gamma(k+alpha+1))
        # Since alpha = -0.5, we can use math.comb if integers, but alpha is not integer.
        # We will use sympy for exact coefficients in the fast path? No, that defeats the purpose.
        # Let's use scipy.special or manual calculation with floats.
        # Actually, for n<=50, float precision is sufficient for *guiding* the search.
        pass
    # We will precompute these outside the function for speed.
    return coeffs

# Precompute Laguerre coefficients for degrees 0, 2, ..., 50 with alpha = -0.5
# L_n^(alpha)(x) = sum_{k=0}^n \binom{n+\alpha}{n-k} \frac{(-x)^k}{k!}
# We need the matrix of coefficients A_lag[k, j] where k is index of L_{2k}, j is power of x.
LAG_COEFFS = []
def _precompute_laguerre():
    from scipy.special import comb
    alpha = -0.5
    for d in range(0, 51, 2):
        coeffs = np.zeros(d + 1)
        for k in range(d + 1):
            # \binom{d+alpha}{d-k}
            # Using gamma: Gamma(d+alpha+1) / (Gamma(d-k+1) * Gamma(k+alpha+1))
            # For d=50, alpha=-0.5, arguments are small integers/half-integers.
            import math
            num = math.lgamma(d + alpha + 1)
            den1 = math.lgamma(d - k + 1)
            den2 = math.lgamma(k + alpha + 1)
            binom_val = math.exp(num - den1 - den2)
            term = binom_val * ((-1)**k) / math.factorial(k)
            coeffs[k] = term
        LAG_COEFFS.append(coeffs)
        
_precompute_laguerre()

def _compute_score_fast(z_list):
    """Fast approximate score using NumPy."""
    z_arr = np.array(z_list, dtype=float)
    if np.any(z_arr <= 0) or np.any(~np.isfinite(z_arr)):
        return float('inf')
    
    # Check distinctness
    z_sorted = np.sort(z_arr)
    if np.any(np.diff(z_sorted) < 1e-8):
        return float('inf')
        
    # Build A matrix (26x26)
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Pre-compute values of L_{2k}(x) and L'_{2k}(x) at 0 and z_i
    # L_n(x) = sum coeffs[j] x^j
    # L'_n(x) = sum j * coeffs[j] x^(j-1)
    
    # Row 0: g(0) = sum c_k L_{2k}(0) = 0
    # L_{2k}(0) = coeffs[0]
    for k in range(26):
        A[0, k] = LAG_COEFFS[k][0]
    b[0] = 0
    
    # Row 1: g'(0) = sum c_k L'_{2k}(0) = 1
    # L'_{2k}(0) = coeffs[1] (since term is c1*x^1, deriv is c1)
    for k in range(26):
        # Derivative at 0 is just the coefficient of x^1
        A[1, k] = LAG_COEFFS[k][1] if len(LAG_COEFFS[k]) > 1 else 0.0
    b[1] = 1
    
    # Rows 2..25
    for i in range(12):
        zi = z_sorted[i]
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        for k in range(26):
            coeffs = LAG_COEFFS[k]
            # Evaluate polynomial at zi
            # Use Horner's rule for speed
            val = 0.0
            dval = 0.0
            # coeffs is ascending power: c0, c1, ..., cn
            # Horner: ((cn*x + c_{n-1})*x + ...)
            # We need value and derivative.
            # Let's just use np.polyval if we reverse coeffs, or manual loop.
            # Manual loop is fine for n<=50.
            val = 0.0
            dval = 0.0
            for j in range(len(coeffs)-1, -1, -1):
                # val = val * x + c_j
                # dval = dval * x + val_old? 
                # Standard derivative evaluation:
                # If P(x) = c_n x^n + ... + c_0
                # P'(x) = n c_n x^{n-1} + ... + c_1
                # We can evaluate both simultaneously.
                pass
            
            # Simpler: use np.polyval on reversed coeffs
            rev_coeffs = coeffs[::-1]
            val = np.polyval(rev_coeffs, zi)
            # Derivative coeffs: j * c_j for x^(j-1)
            deriv_coeffs = np.zeros(len(coeffs))
            for j in range(1, len(coeffs)):
                deriv_coeffs[j-1] = j * coeffs[j]
            # Wait, deriv_coeffs is for x^0 ... x^{n-1}
            # So we can use np.polyval(deriv_coeffs[::-1], zi)?
            # No, deriv_coeffs[k] corresponds to x^k.
            # So we just evaluate the polynomial defined by deriv_coeffs.
            dval = np.polyval(deriv_coeffs[::-1], zi)
            
            A[row_g, k] = val
            A[row_gp, k] = dval
        b[row_g] = 0
        b[row_gp] = 0
        
    try:
        c = np.linalg.solve(A, b)
    except Exception:
        return float('inf')
        
    # Construct g(x) coefficients in standard basis
    # g(x) = sum_k c_k L_{2k}(x)
    # L_{2k}(x) has degree 2k. Max degree 50.
    g_coeffs = np.zeros(51)
    for k in range(26):
        g_coeffs[:len(LAG_COEFFS[k])] += c[k] * LAG_COEFFS[k]
        
    # Construct denominator D(x) = x * prod (x - z_i)^2
    # Start with [1, 0] for x
    D = np.array([0.0, 1.0])
    for zi in z_sorted:
        # (x - zi)^2 = x^2 - 2 zi x + zi^2
        sq = np.array([zi*zi, -2*zi, 1.0])
        D = np.convolve(D, sq)
        
    # Divide g(x) by D(x)
    # g_coeffs is ascending, np.polydiv expects descending.
    g_desc = g_coeffs[::-1]
    D_desc = D[::-1]
    
    try:
        q_desc, _ = np.polydiv(g_desc, D_desc)
    except Exception:
        return float('inf')
        
    # q_desc is descending coeffs of q(x)
    # Find roots
    try:
        roots = np.roots(q_desc)
    except Exception:
        return float('inf')
        
    # Filter real roots
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    
    # Check sign change for each real root
    # q(x) = sum q_desc[i] x^(n-i)
    # We need to check q(r - eps) and q(r + eps)
    eps = 1e-10 # Use float eps for fast check
    
    max_r = 0.0
    for r in real_roots:
        q_minus = np.polyval(q_desc, r - eps)
        q_plus = np.polyval(q_desc, r + eps)
        if q_minus * q_plus < 0:
            if r > max_r:
                max_r = r
                
    B = max(0.0, max_r)
    score = B / (2 * math.pi)
    return score

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z_list(z_list):
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    z_sorted = sorted(z_list)
    for i in range(11):
        if z_sorted[i+1] - z_sorted[i] < 1e-6:
            return False
    return True

def construct():
    deadline = time.time() + 580.0 # 50s margin
    
    # Initial candidates
    candidates = []
    
    # 1. Small integers
    candidates.append([float(i) for i in range(1, 13)])
    
    # 2. Geometric
    candidates.append([1.0 * (1.2 ** i) for i in range(12)])
    
    # 3. Bessel (large, for comparison)
    candidates.append([_bessel_zero_J0(i)**2 for i in range(1, 13)])
    
    # 4. Random small
    np.random.seed(42)
    candidates.append(np.sort(np.random.uniform(0.1, 10.0, 12)).tolist())
    
    # Fast evaluate all
    best_fast_score = float('inf')
    best_z = candidates[0]
    
    for z in candidates:
        if not _validate_z_list(z):
            continue
        score = _compute_score_fast(z)
        if score < best_fast_score:
            best_fast_score = score
            best_z = z
            
    # Local refinement using scipy.optimize
    if _time_remaining(deadline) > 100:
        # Define objective for minimization
        def obj(z_flat):
            z_list = list(z_flat)
            if not _validate_z_list(z_list):
                return 1e10
            return _compute_score_fast(z_list)
            
        # Start from best_z
        x0 = np.array(best_z)
        
        # Try Nelder-Mead
        try:
            res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if res.success and _validate_z_list(res.x):
                fast_score = _compute_score_fast(res.x)
                if fast_score < best_fast_score:
                    best_fast_score = fast_score
                    best_z = res.x.tolist()
        except Exception:
            pass

    # Final exact verification of the best candidate found by fast search
    best_exact_score = float('inf')
    final_z = best_z
    
    if _time_remaining(deadline) > 60:
        try:
            exact_score = _compute_score_exact(best_z)
            if exact_score is not None:
                best_exact_score = exact_score
                final_z = best_z
        except Exception:
            pass
            
    # If we have time, try a few perturbations of the best exact candidate
    if _time_remaining(deadline) > 120 and best_exact_score < float('inf'):
        # Perturb each z_i
        for i in range(12):
            if _time_remaining(deadline) < 60:
                break
            for delta in [0.1, -0.1, 0.5, -0.5]:
                if _time_remaining(deadline) < 60:
                    break
                cand = list(best_z)
                cand[i] = best_z[i] + delta
                if not _validate_z_list(cand):
                    continue
                try:
                    sc = _compute_score_exact(cand)
                    if sc is not None and sc < best_exact_score:
                        best_exact_score = sc
                        final_z = cand
                except Exception:
                    pass

    # If no exact score was found, fall back to the best fast candidate (assuming it's valid)
    if best_exact_score == float('inf'):
        final_z = best_z
        # We should probably return a known valid set if the best fast one fails exact check
        # But the task says "invalid result ... scores 0". 
        # So we must ensure the returned list is valid.
        if not _validate_z_list(final_z):
             final_z = [float(i) for i in range(1, 13)]

    return final_z
# EVOLVE-BLOCK-END
