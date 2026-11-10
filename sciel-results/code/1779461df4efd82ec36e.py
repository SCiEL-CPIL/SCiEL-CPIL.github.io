"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_genlaguerre, genlaguerre

# EVOLVE-BLOCK-START

# Precompute Laguerre polynomials L_0^(-1/2), L_2^(-1/2), ..., L_50^(-1/2)
# and their derivatives for faster evaluation in the objective function.
_LAGUERRE_ALPHA = -0.5
_LAGUERRE_POLYS = [genlaguerre(2*k, _LAGUERRE_ALPHA) for k in range(26)]
_LAGUERRE_DERIVS = [p.deriv() for p in _LAGUERRE_POLYS]

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
    
    # Build A and b using precomputed polynomials
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Row 0: g(0) = 0
    for k in range(26):
        A[0, k] = _LAGUERRE_POLYS[k](0)
    b[0] = 0
    
    # Row 1: g'(0) = 1
    for k in range(26):
        A[1, k] = _LAGUERRE_DERIVS[k](0)
    b[1] = 1
    
    # Rows 2..25
    for i in range(12):
        zi = z[i]
        row_idx = 2 + 2*i
        for k in range(26):
            A[row_idx, k] = _LAGUERRE_POLYS[k](zi)
        b[row_idx] = 0
        
        row_idx = 3 + 2*i
        for k in range(26):
            A[row_idx, k] = _LAGUERRE_DERIVS[k](zi)
        b[row_idx] = 0
            
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e100
        
    # Construct g(x)
    g = np.poly1d([0]) 
    for k in range(26):
        g = g + c[k] * _LAGUERRE_POLYS[k]
        
    # Construct D(x) = x * prod (x - z_i)^2
    D = np.poly1d([1, 0]) # x
    for zi in z:
        D = D * (np.poly1d([1, -zi]) ** 2)
        
    # q = g / D
    try:
        q, rem = np.polydiv(g, D)
    except Exception:
        return 1e100
        
    # Find roots of q
    # q is a polynomial of degree 25
    try:
        roots = np.roots(q)
    except Exception:
        return 1e100
        
    # Filter for real roots (small imaginary part)
    real_roots = []
    for root in roots:
        if abs(root.imag) < 1e-8:
            real_roots.append(root.real)
            
    if len(real_roots) == 0:
        return 0.0
    return np.max(real_roots)

def construct():
    """Return 12 double-root locations z_1..z_12 using spectral interpolation strategy."""
    deadline = time.time() + 540.0  # Keep 60s margin
    
    # Define initial guesses
    # 1. Roots of L_12^(-1/2)
    try:
        z_init_1 = roots_genlaguerre(12, -0.5)
        z_init_1 = [float(z) for z in z_init_1]
    except Exception:
        z_init_1 = [4.0 * (i + 1) for i in range(12)]
        
    # 2. Roots of L_12^(0)
    try:
        z_init_2 = roots_genlaguerre(12, 0)
        z_init_2 = [float(z) for z in z_init_2]
    except Exception:
        z_init_2 = [4.0 * (i + 1) for i in range(12)]
        
    # 3. Uniformly spaced
    z_init_3 = [1.0 * (i + 1) for i in range(12)]
    
    initial_guesses = [z_init_1, z_init_2, z_init_3]
    
    best_z = z_init_1
    best_score = _float_score(z_init_1)
    
    # Optimize from each initial guess
    for z_init in initial_guesses:
        if time.time() > deadline - 60:
            break
            
        def objective(z_vec):
            return _float_score(z_vec)
            
        # Bounds: z_i > 1e-6
        bounds = [(1e-6, None) for _ in range(12)]
        
        try:
            # Use Powell method
            res = minimize(objective, z_init, method='Powell', bounds=bounds, 
                           options={'maxiter': 300, 'maxfev': 10000})
            if res.fun < best_score:
                best_score = res.fun
                best_z = res.x
        except Exception:
            pass
            
        # Also try Nelder-Mead from the best found in this branch
        if time.time() < deadline - 30:
            try:
                res2 = minimize(objective, best_z, method='Nelder-Mead', 
                                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8})
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
