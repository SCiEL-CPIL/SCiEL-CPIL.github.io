"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

def _compute_score_exact(z_list):
    """Compute B/(2*pi) for given z_list using sympy exact arithmetic."""
    from sympy import symbols, Rational, Poly, diff, real_roots, pi as sympy_pi, Matrix, binomial
    import sympy
    
    z_rationals = sorted(Rational(float(z)) for z in z_list)
    
    x = symbols('x')
    alpha = Rational(-1, 2)
    
    # Build generalized Laguerre polynomials L_0^(alpha), L_2^(alpha), ..., L_50^(alpha)
    # L_n^(alpha)(x) = sum_{k=0}^{n} binomial(n+alpha, n-k) * (-x)^k / k!
    def L_gen_explicit(n, alpha_val, x_sym):
        result = 0
        for k in range(n + 1):
            coeff = binomial(n + alpha_val, n - k) * (-1)**k
            result = result + coeff * x_sym**k / sympy.factorial(k)
        return result
    
    # Build the 26 Laguerre polynomials for degrees 0, 2, 4, ..., 50
    laguerre_pols = []
    for d in range(0, 51, 2):
        laguerre_pols.append(L_gen_explicit(d, alpha, x))
    
    # Precompute derivatives
    laguerre_pols_deriv = [p.diff(x) for p in laguerre_pols]
    
    # Build system: g(x) = sum_{k=0}^{25} c_k * L_{2k}^(alpha)(x)
    # Conditions: g(0)=0, g'(0)=1, g(z_i)=0, g'(z_i)=0 for i=1..12
    # Total: 2 + 12*2 = 26 equations, 26 unknowns c_0..c_25
    
    A = Matrix(26, 26)
    b = Matrix(26, 1)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = laguerre_pols[k].subs(x, 0)
    b[0, 0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = laguerre_pols_deriv[k].subs(x, 0)
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
            A[row_idx, k] = laguerre_pols_deriv[k].subs(x, zi)
        b[row_idx, 0] = 0
    
    # Solve
    try:
        c = A.LUsolve(b)
    except Exception:
        return None
    
    # Build g(x)
    g = sum(c[k] * laguerre_pols[k] for k in range(26))
    
    # Build denominator: x * prod_i (x - z_i)^2
    denom = x
    for zi in z_rationals:
        denom = denom * (x - zi)**2
    
    # q(x) = g(x) / denom via exact polynomial division
    g_poly = Poly(g, x)
    denom_poly = Poly(denom, x)
    
    try:
        q_poly, remainder = g_poly.div(denom_poly)
    except Exception:
        return None
    
    # Find real roots of q
    q_expr = q_poly.as_expr()
    try:
        real_roots_list = real_roots(q_expr)
    except Exception:
        return None
    
    # Count roots where q has strictly opposite signs at r-1e-198 and r+1e-198
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

def _float_score(z_list):
    """Fast floating point approximation of the score."""
    z = np.array(z_list, dtype=float)
    if np.any(z <= 0) or np.any(~np.isfinite(z)):
        return 1e100
    # Check distinctness
    for i in range(12):
        for j in range(i+1, 12):
            if abs(z[i] - z[j]) < 1e-9:
                return 1e100
    
    # Precompute Laguerre polynomials L_0, L_2, ..., L_50 with alpha=-0.5
    alphas = -0.5
    L_pols = [genlaguerre(2*k, alphas) for k in range(26)]
    
    # Build A and b
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = L_pols[k](0)
    b[0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = L_pols[k].deriv()(0)
    b[1] = 1
    
    # Rows 2..25
    for i in range(12):
        zi = z[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = L_pols[k](zi)
        b[row_idx, 0] = 0
        
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = L_pols[k].deriv()(zi)
        b[row_idx, 0] = 0
            
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e100
        
    # Construct g(x)
    g = np.poly1d([0]) 
    for k in range(26):
        g = g + c[k] * L_pols[k]
        
    # Construct D(x) = x * prod (x - z_i)^2
    D = np.poly1d([1, 0]) # x
    for zi in z:
        D = D * (np.poly1d([1, -zi]) ** 2)
        
    # q = g / D
    try:
        q, rem = np.polydiv(g, D)
    except Exception:
        return 1e100
        
    roots = np.roots(q)
    real_roots = roots[np.isreal(roots)].real
    if len(real_roots) == 0:
        return 0.0
    return np.max(real_roots)

def construct():
    """Return 12 double-root locations z_1..z_12 using spectral interpolation strategy."""
    deadline = time.time() + 540.0  # Keep 60s margin
    
    # Generate multiple initial guesses based on spectral properties
    initial_guesses = []
    
    # 1. Roots of L_12^(-0.5)(x)
    try:
        z_init1 = roots_genlaguerre(12, -0.5)
        initial_guesses.append([float(z) for z in z_init1])
    except Exception:
        pass
        
    # 2. Roots of L_12^(0)(x)
    try:
        z_init2 = roots_genlaguerre(12, 0)
        initial_guesses.append([float(z) for z in z_init2])
    except Exception:
        pass
        
    # 3. Scaled roots of L_12^(-0.5)(x)
    try:
        z_init3 = roots_genlaguerre(12, -0.5)
        initial_guesses.append([float(z) * 0.8 for z in z_init3])
        initial_guesses.append([float(z) * 1.2 for z in z_init3])
    except Exception:
        pass
        
    # 4. Uniform distribution in [1, 50]
    initial_guesses.append(list(np.linspace(1, 50, 12)))
    
    # 5. Geometric distribution
    initial_guesses.append(list(np.geomspace(1, 50, 12)))
    
    best_z = initial_guesses[0]
    best_score = _float_score(best_z)
    
    bounds = [(1e-6, None) for _ in range(12)]
    
    for z_start in initial_guesses:
        if time.time() > deadline - 30:
            break
            
        # Ensure valid start
        z_start = np.array(z_start, dtype=float)
        if np.any(z_start <= 0) or np.any(~np.isfinite(z_start)):
            continue
            
        # Use L-BFGS-B for better convergence on smooth parts
        try:
            res = minimize(_float_score, z_start, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 200, 'maxls': 50, 'ftol': 1e-9})
            if res.fun < best_score:
                best_score = res.fun
                best_z = res.x
        except Exception:
            pass
            
        # Refine with SLSQP if time permits
        if time.time() < deadline - 60:
            try:
                res2 = minimize(_float_score, best_z, method='SLSQP', bounds=bounds,
                                options={'maxiter': 100, 'ftol': 1e-10})
                if res2.fun < best_score:
                    best_score = res2.fun
                    best_z = res2.x
            except Exception:
                pass

    # Final validation
    best_z = [float(z) for z in best_z]
    for i in range(12):
        if not math.isfinite(best_z[i]) or best_z[i] <= 0:
            best_z = [4.0 * (i + 1) for i in range(12)]
            break
    for i in range(12):
        for j in range(i+1, 12):
            if abs(best_z[i] - best_z[j]) < 1e-12:
                best_z = [4.0 * (i + 1) for i in range(12)]
                break
                
    return best_z

# EVOLVE-BLOCK-END
