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
    denom_poly = Poly(denom, x)
    
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

def construct():
    """
    Return 12 double-root locations z_1..z_12.
    Uses numerical optimization to find a good configuration, then verifies with exact arithmetic.
    """
    deadline = time.time() + 570.0
    
    bounds = [(0.01, 50.0)] * 12
    
    best_numeric_result = None
    best_numeric_score = float('inf')
    
    def objective(z):
        return _score_numeric(z)
        
    try:
        res = differential_evolution(objective, bounds, maxiter=100, popsize=15, seed=42, polish=False, workers=1)
        if res.success or res.fun < best_numeric_score:
            best_numeric_score = res.fun
            best_numeric_result = res.x
    except Exception:
        pass
        
    if best_numeric_result is None or best_numeric_score > 10.0:
        for _ in range(50):
            z_rand = np.random.uniform(0.1, 20.0, 12)
            z_rand = np.sort(z_rand)
            if np.all(np.diff(z_rand) > 1e-3):
                s = _score_numeric(z_rand)
                if s < best_numeric_score:
                    best_numeric_score = s
                    best_numeric_result = z_rand
                    if _time_remaining(deadline) < 120:
                        break
                        
    if best_numeric_result is not None and _time_remaining(deadline) > 120:
        try:
            res_nm = minimize(objective, best_numeric_result, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res_nm.fun < best_numeric_score:
                best_numeric_score = res_nm.fun
                best_numeric_result = res_nm.x
        except Exception:
            pass
            
    candidates = []
    if best_numeric_result is not None:
        candidates.append(best_numeric_result)
        
    for _ in range(5):
        if best_numeric_result is not None:
            z_pert = best_numeric_result * (1 + np.random.uniform(-0.1, 0.1, 12))
            z_pert = np.sort(z_pert)
            if np.all(z_pert > 0) and np.all(np.diff(z_pert) > 1e-4):
                candidates.append(z_pert)
                
    best_exact_score = float('inf')
    best_exact_z = None
    
    for cand in candidates:
        if _time_remaining(deadline) < 90:
            break
        try:
            score = _compute_score_exact(cand)
            if score is not None and score < best_exact_score:
                best_exact_score = score
                best_exact_z = list(cand)
        except Exception:
            continue
            
    if best_exact_z is None:
        return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        
    return best_exact_z
# EVOLVE-BLOCK-END
