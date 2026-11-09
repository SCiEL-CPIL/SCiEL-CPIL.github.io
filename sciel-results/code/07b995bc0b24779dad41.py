"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def _bessel_zero_J0(k):
    """Approximate k-th zero of J_0 using asymptotic expansion (high accuracy)."""
    x = (k - 0.25) * math.pi
    if x < 0.1:
        return 2.4048255576957728  # j_{0,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

def _bessel_zero_J1(k):
    """Approximate k-th zero of J_1 using asymptotic expansion."""
    x = (k + 0.25) * math.pi
    if x < 0.1:
        return 3.8317059702075125  # j_{1,1} exact
    return x - 1.0/(8.0*x) + 9.0/(128.0*x**3) - 225.0/(3072.0*x**5)

# Cache for Laguerre polynomials to avoid recomputation
_laguerre_cache = {}

def _get_laguerre_pols_cached():
    """Get cached Laguerre polynomials for degrees 0,2,...,50 with alpha=-1/2."""
    if _laguerre_cache:
        return _laguerre_cache['pols'], _laguerre_cache['x'], _laguerre_cache['alpha']
    
    from sympy import symbols, Rational, laguerre
    x = symbols('x')
    alpha = Rational(-1, 2)
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(laguerre(d, x, alpha))
    
    _laguerre_cache['pols'] = laguerre_pols
    _laguerre_cache['x'] = x
    _laguerre_cache['alpha'] = alpha
    return laguerre_pols, x, alpha

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import Rational, Poly, real_roots, pi as sympy_pi, Matrix
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    laguerre_pols, x, alpha = _get_laguerre_pols_cached()
    
    # Precompute values and derivatives at 0
    L_at_0 = [p.subs(x, 0) for p in laguerre_pols]
    Lp_at_0 = [p.diff(x).subs(x, 0) for p in laguerre_pols]
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = L_at_0[k]
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = Lp_at_0[k]
    b[1, 0] = 1
    
    # Rows 2..25: for each z_i, g(z_i)=0 and g'(z_i)=0
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

def _time_remaining(deadline):
    return deadline - time.time()

def _validate_z_list(z_list):
    """Check that z_list is valid: finite, > 0, pairwise distinct."""
    if len(z_list) != 12:
        return False
    for z in z_list:
        if not math.isfinite(z) or z <= 0:
            return False
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z_list[i] - z_list[j]) < 1e-10:
                return False
    return True

def _quick_numeric_estimate(z_list):
    """
    Quick numerical estimate of B/(2*pi) using float64 to pre-filter candidates.
    This is fast (<1s) and gives a rough ranking.
    """
    try:
        from scipy.special import eval_genlaguerre
        from scipy.optimize import brentq
        import numpy as np
        
        z_arr = np.array(z_list, dtype=np.float64)
        alpha = -0.5
        
        # Build the 26 Laguerre polynomials numerically
        # L_d^(alpha)(x) for d = 0, 2, ..., 50
        degrees = list(range(0, 51, 2))
        
        # Evaluate at 0 and derivatives
        L_at_0 = np.array([eval_genlaguerre(d, alpha, 0.0) for d in degrees])
        Lp_at_0 = np.array([eval_genlaguerre(d, alpha, 0.0, 1) for d in degrees])
        
        # Build matrix A and vector b
        A = np.zeros((26, 26))
        b = np.zeros(26)
        
        A[0, :] = L_at_0
        b[0] = 0.0
        A[1, :] = Lp_at_0
        b[1] = 1.0
        
        for i in range(12):
            zi = z_arr[i]
            row_idx = 2 + 2*i
            A[row_idx, :] = [eval_genlaguerre(d, alpha, zi) for d in degrees]
            b[row_idx] = 0.0
            row_idx = 3 + 2*i
            A[row_idx, :] = [eval_genlaguerre(d, alpha, zi, 1) for d in degrees]
            b[row_idx] = 0.0
        
        try:
            c = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None
        
        # Build g(x) and q(x)
        def g_func(x_val):
            return sum(c[k] * eval_genlaguerre(degrees[k], alpha, x_val) for k in range(26))
        
        def g_deriv_func(x_val):
            return sum(c[k] * eval_genlaguerre(degrees[k], alpha, x_val, 1) for k in range(26))
        
        # q(x) = g(x) / (x * prod_i (x - z_i)^2)
        def q_func(x_val):
            if x_val == 0:
                return 0.0
            denom = x_val
            for zi in z_arr:
                denom *= (x_val - zi)**2
            if abs(denom) < 1e-30:
                return 0.0
            return g_func(x_val) / denom
        
        # Find real roots of q(x) in a reasonable range
        # q(x) is a polynomial of degree 50 - 25 = 25
        # Look for roots in (0, max_z * 3)
        max_z = max(z_arr)
        search_max = max_z * 3.0
        
        # Sample to find sign changes
        n_samples = 200
        x_samples = np.linspace(1e-6, search_max, n_samples)
        q_samples = np.array([q_func(x) for x in x_samples])
        
        # Find sign changes
        sign_changes = []
        for i in range(n_samples - 1):
            if q_samples[i] * q_samples[i+1] < 0:
                # Refine with brentq
                try:
                    root = brentq(q_func, x_samples[i], x_samples[i+1], xtol=1e-12)
                    sign_changes.append(root)
                except Exception:
                    pass
        
        if not sign_changes:
            return 0.0
        
        # Check which roots have opposite signs at r±1e-198 (use smaller eps for float)
        eps = 1e-15
        valid_roots = []
        for r in sign_changes:
            q_minus = q_func(r - eps)
            q_plus = q_func(r + eps)
            if q_minus * q_plus < 0:
                valid_roots.append(r)
        
        if not valid_roots:
            return 0.0
        
        B = max(0.0, max(valid_roots))
        return B / (2.0 * math.pi)
    
    except Exception:
        return None

def _try_local_search(base_z, deadline, best_score, best_z):
    """Try local perturbations around base_z, update best if improved."""
    # Try scaling the entire configuration
    scale_factors = [0.8, 0.9, 1.1, 1.2, 0.7, 1.3]
    for sf in scale_factors:
        if _time_remaining(deadline) < 100:
            break
        candidate = [z * sf for z in base_z]
        if not _validate_z_list(candidate):
            continue
        score = _compute_score_exact(candidate)
        if score is not None and score < best_score:
            best_score = score
            best_z = candidate
    
    # Try individual perturbations in log-space
    for i in range(12):
        if _time_remaining(deadline) < 100:
            break
        for delta_log in [0.1, -0.1, 0.2, -0.2, 0.05, -0.05]:
            if _time_remaining(deadline) < 100:
                break
            candidate = list(best_z)
            candidate[i] = best_z[i] * math.exp(delta_log)
            if not _validate_z_list(candidate):
                continue
            score = _compute_score_exact(candidate)
            if score is not None and score < best_score:
                best_score = score
                best_z = candidate
    
    return best_score, best_z

def construct():
    """Return 12 double-root locations z_1..z_12 using Bessel-zero-inspired parameterization."""
    # 630s kill. Set deadline to 580s to allow for cleanup and return.
    deadline = time.time() + 580.0
    
    # Generate multiple candidate initial configurations
    candidates = []
    
    # 1. J_0 zeros squared (commonly good for Laguerre-related roots)
    z_j0 = [_bessel_zero_J0(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j0):
        candidates.append(('j0_sq', z_j0))
    
    # 2. J_1 zeros squared
    z_j1 = [_bessel_zero_J1(i)**2 for i in range(1, 13)]
    if _validate_z_list(z_j1):
        candidates.append(('j1_sq', z_j1))
    
    # 3. Geometric progression with ratio 1.5
    geo_ratio = 1.5
    z_geo = [2.0 * geo_ratio**i for i in range(12)]
    if _validate_z_list(z_geo):
        candidates.append(('geo_1.5', z_geo))
    
    # 4. Geometric progression with ratio 1.3
    geo_ratio = 1.3
    z_geo2 = [1.0 * geo_ratio**i for i in range(12)]
    if _validate_z_list(z_geo2):
        candidates.append(('geo_1.3', z_geo2))
    
    # 5. Arithmetic progression
    z_arith = [2.0 + 3.0*i for i in range(12)]
    if _validate_z_list(z_arith):
        candidates.append(('arith', z_arith))
    
    # 6. J_0 zeros (not squared)
    z_j0_raw = [_bessel_zero_J0(i) for i in range(1, 13)]
    if _validate_z_list(z_j0_raw):
        candidates.append(('j0_raw', z_j0_raw))
    
    # 7. Logarithmic spacing
    z_log = [0.5 * math.exp(0.3 * i) for i in range(12)]
    if _validate_z_list(z_log):
        candidates.append(('log', z_log))
        
    # Default fallback if no candidates are valid (should not happen)
    default_z = [4.0 * (i+1) for i in range(12)]
    if not candidates:
        candidates.append(('default', default_z))

    # Quick numeric pre-filter to rank candidates
    ranked_candidates = []
    for name, z_list in candidates:
        est = _quick_numeric_estimate(z_list)
        if est is not None:
            ranked_candidates.append((est, name, z_list))
        else:
            ranked_candidates.append((float('inf'), name, z_list))
    
    # Sort by estimated score (lower is better)
    ranked_candidates.sort(key=lambda x: x[0])
    
    best_score = float('inf')
    best_z = candidates[0][1]
    
    # Evaluate top candidates with exact computation
    for est_score, name, z_list in ranked_candidates:
        if _time_remaining(deadline) < 150:
            break
        try:
            score = _compute_score_exact(z_list)
            if score is not None:
                if score < best_score:
                    best_score = score
                    best_z = z_list
        except Exception:
            pass
    
    # If we have time, try local search from the best solution
    if _time_remaining(deadline) > 200 and best_score < float('inf'):
        try:
            best_score, best_z = _try_local_search(best_z, deadline, best_score, best_z)
        except Exception:
            pass
    
    # Final validation
    if not _validate_z_list(best_z):
        best_z = default_z
    
    return best_z
# EVOLVE-BLOCK-END
