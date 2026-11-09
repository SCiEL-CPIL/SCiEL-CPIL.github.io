"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

_L_coeffs_cache = None

def _get_laguerre_coeffs():
    global _L_coeffs_cache
    if _L_coeffs_cache is None:
        _L_coeffs_cache = []
        alpha = -0.5
        max_deg = 50
        for k in range(26):
            n = 2*k
            L_n = genlaguerre(n, alpha)
            c = np.asarray(L_n.c, dtype=float)
            if len(c) < max_deg + 1:
                c = np.pad(c, (0, max_deg + 1 - len(c)), mode='constant')
            _L_coeffs_cache.append(c)
    return _L_coeffs_cache

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix
    from sympy.polys.orthopolys import genlaguerre as sympy_genlaguerre
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(sympy_genlaguerre(d, alpha, x))
    
    A = Matrix(26, 26, lambda i, j: 0)
    b = Matrix(26, 1, lambda i, j: 0)
    
    L0_vals = [poly.subs(x, 0) for poly in laguerre_pols]
    L0_deriv_vals = [poly.diff(x).subs(x, 0) for poly in laguerre_pols]
    
    for k in range(26):
        A[0, k] = L0_vals[k]
    b[0, 0] = 0
    
    for k in range(26):
        A[1, k] = L0_deriv_vals[k]
    b[1, 0] = 1
    
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
    denom_poly = Poly(denom)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
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

def _score_numeric(z_list):
    """Fast numerical approximation of the score for optimization."""
    z = np.array(z_list, dtype=float)
    if np.any(z <= 0) or np.any(~np.isfinite(z)):
        return 1e10
    z_sorted = np.sort(z)
    if np.any(np.diff(z_sorted) < 1e-8):
        return 1e10
        
    alpha = -0.5
    max_deg = 50
    coeffs_L = _get_laguerre_coeffs()
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    for k in range(26):
        A[0, k] = coeffs_L[k][-1]
    b[0] = 0
    
    for k in range(26):
        if k == 0:
            A[1, k] = 0.0
        else:
            n = 2*k - 1
            alpha_deriv = 0.5
            L_deriv = genlaguerre(n, alpha_deriv)
            A[1, k] = -L_deriv.c[-1]
    b[1] = 1.0
    
    for i in range(12):
        zi = z[i]
        row_g = 2 + 2*i
        row_dg = 3 + 2*i
        
        for k in range(26):
            A[row_g, k] = eval_genlaguerre(2*k, alpha, zi)
            b[row_g] = 0.0
            
            if k == 0:
                A[row_dg, k] = 0.0
            else:
                n = 2*k - 1
                alpha_deriv = 0.5
                A[row_dg, k] = -eval_genlaguerre(n, alpha_deriv, zi)
            b[row_dg] = 0.0
            
    try:
        c_coeffs = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    g_coeffs = np.zeros(max_deg + 1)
    for k in range(26):
        g_coeffs += c_coeffs[k] * coeffs_L[k]
        
    D_coeffs = np.array([1.0, 0.0])
    for zi in z:
        quad = np.array([1.0, -2.0*zi, zi**2])
        D_coeffs = np.convolve(D_coeffs, quad)
        
    try:
        q_coeffs, _ = np.polydiv(g_coeffs, D_coeffs)
    except Exception:
        return 1e10
        
    roots = np.roots(q_coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-9]
    
    if not real_roots:
        return 0.0
        
    q_poly = np.poly1d(q_coeffs)
    q_prime_poly = np.polyder(q_poly)
    
    max_r = 0.0
    for r in real_roots:
        if abs(q_prime_poly(r)) > 1e-6:
            if r > max_r:
                max_r = r
                
    B = max(0.0, max_r)
    score = B / (2 * np.pi)
    return score

def _time_remaining(deadline):
    return deadline - time.time()

def _generate_structured_candidates():
    """Generate structured z configurations using various patterns."""
    candidates = []
    
    # Log-spaced
    for n_points in [12]:
        for start_log in np.linspace(-2, 1, 8):
            for end_log in np.linspace(0.5, 2.5, 8):
                if end_log <= start_log:
                    continue
                z = np.logspace(start_log, end_log, n_points)
                if np.all(z > 0) and np.all(np.diff(z) > 1e-4):
                    candidates.append(z.copy())
    
    # Arithmetic with varying step
    for start in np.linspace(0.1, 5.0, 6):
        for step in np.linspace(0.1, 3.0, 8):
            z = start + np.arange(12) * step
            if np.all(z > 0) and np.all(np.diff(z) > 1e-4):
                candidates.append(z.copy())
    
    # Power law
    for p in np.linspace(0.5, 2.0, 6):
        for base in np.linspace(0.5, 5.0, 6):
            z = base * (np.arange(1, 13) ** p)
            if np.all(z > 0) and np.all(np.diff(z) > 1e-4):
                candidates.append(z.copy())
    
    # Interleaved: alternate between two scales
    for scale1 in np.linspace(0.5, 3.0, 5):
        for scale2 in np.linspace(3.0, 10.0, 5):
            if scale2 <= scale1:
                continue
            z = np.empty(12)
            for i in range(12):
                if i % 2 == 0:
                    z[i] = scale1 * (i // 2 + 1)
                else:
                    z[i] = scale2 * ((i + 1) // 2)
            z = np.sort(z)
            if np.all(z > 0) and np.all(np.diff(z) > 1e-4):
                candidates.append(z.copy())
    
    # Clustered: groups of 4 with gaps
    for gap in np.linspace(2.0, 10.0, 5):
        for within in np.linspace(0.1, 0.5, 4):
            z = []
            for g in range(3):
                start = gap * (g + 1)
                for w in range(4):
                    z.append(start + w * within)
            z = np.array(z)
            if np.all(z > 0) and np.all(np.diff(z) > 1e-4):
                candidates.append(z.copy())
    
    return candidates

def _local_refine(z_init, deadline):
    """Refine a candidate using coordinate descent."""
    z = z_init.copy()
    best_score = _score_numeric(z)
    
    while _time_remaining(deadline) > 60:
        improved = False
        for i in range(12):
            # Try moving z[i]
            for delta_factor in [0.9, 0.95, 1.05, 1.1]:
                z_new = z.copy()
                z_new[i] = z[i] * delta_factor
                z_new = np.sort(z_new)
                if np.all(z_new > 0) and np.all(np.diff(z_new) > 1e-6):
                    score_new = _score_numeric(z_new)
                    if score_new < best_score - 1e-10:
                        z = z_new
                        best_score = score_new
                        improved = True
                        break
        if not improved:
            # Try smaller steps
            for i in range(12):
                for delta_factor in [0.98, 1.02]:
                    z_new = z.copy()
                    z_new[i] = z[i] * delta_factor
                    z_new = np.sort(z_new)
                    if np.all(z_new > 0) and np.all(np.diff(z_new) > 1e-6):
                        score_new = _score_numeric(z_new)
                        if score_new < best_score - 1e-10:
                            z = z_new
                            best_score = score_new
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    return z, best_score

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Uses structured candidate generation with local refinement.
    """
    deadline = time.time() + 570.0
    
    # Generate structured candidates
    candidates = _generate_structured_candidates()
    
    # Also add some random candidates
    np.random.seed(42)
    for _ in range(30):
        z = np.sort(np.random.uniform(0.1, 20.0, 12))
        if np.all(z > 0) and np.all(np.diff(z) > 1e-3):
            candidates.append(z)
    
    # Score all candidates numerically
    scored = []
    for cand in candidates:
        if _time_remaining(deadline) < 120:
            break
        try:
            score = _score_numeric(cand)
            scored.append((score, cand))
        except Exception:
            continue
    
    # Sort by score
    scored.sort(key=lambda x: x[0])
    
    # Take top candidates and refine
    best_exact_score = float('inf')
    best_exact_z = None
    
    top_n = min(10, len(scored))
    for i in range(top_n):
        if _time_remaining(deadline) < 90:
            break
        
        score, z = scored[i]
        
        # Refine locally
        try:
            z_refined, score_refined = _local_refine(z, deadline)
            if score_refined < score:
                z = z_refined
                score = score_refined
        except Exception:
            pass
        
        # Verify with exact arithmetic
        if _time_remaining(deadline) > 60:
            try:
                exact_score = _compute_score_exact(z)
                if exact_score is not None and exact_score < best_exact_score:
                    best_exact_score = exact_score
                    best_exact_z = list(z)
            except Exception:
                continue
    
    if best_exact_z is None:
        # Fallback: use the best numerical candidate
        if scored:
            best_exact_z = list(scored[0][1])
        else:
            best_exact_z = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    
    return best_exact_z
# EVOLVE-BLOCK-END
