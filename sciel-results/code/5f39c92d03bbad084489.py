import numpy as np
from scipy.special import roots_genlaguerre, eval_genlaguerre
from scipy.optimize import minimize
import time

def _laguerre_coeffs(n, alpha):
    """
    Compute coefficients of L_n^(alpha)(x) in the basis [1, x, x^2, ...].
    L_n^(alpha)(x) = sum_{j=0}^n (-1)^j * binom(n+alpha, n-j) * x^j / j!
    """
    coeffs = np.zeros(n + 1)
    from scipy.special import binom, factorial
    for j in range(n + 1):
        coeffs[j] = ((-1) ** j) * binom(n + alpha, n - j) / factorial(j)
    return coeffs

def _surrogate_score(zs):
    n_nodes = len(zs)
    alpha = -0.5
    
    # Precompute Laguerre coefficients for L_0, L_2, ..., L_50
    lag_coeffs = []
    for k in range(26):
        n = 2 * k
        c = _laguerre_coeffs(n, alpha)
        lag_coeffs.append(c)
        
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # g(0) = 0
    for k in range(26):
        A[0, k] = lag_coeffs[k][0]
    b[0] = 0.0
    
    # g'(0) = 1
    # Derivative at 0:
    # L_n'(0) = -binom(n+alpha, n-1) for n>=1, 0 for n=0.
    # Or just use the coefficient of x in L_n, which is L_n'(0).
    for k in range(26):
        if 2*k >= 1:
            A[1, k] = lag_coeffs[k][1]
        else:
            A[1, k] = 0.0
    b[1] = 1.0
    
    # g(z_i) = 0, g'(z_i) = 0
    for i, z in enumerate(zs):
        row_g = 2 + 2*i
        row_gp = 3 + 2*i
        for k in range(26):
            # Evaluate L_{2k}(z)
            # We can use the polynomial coefficients we already have
            # np.polyval(lag_coeffs[k][::-1], z)
            A[row_g, k] = np.polyval(lag_coeffs[k][::-1], z)
            
            # Derivative
            # np.polyder then polyval
            if 2*k >= 1:
                # lag_coeffs[k] is [c0, c1, ...]
                # Derivative coeffs: [c1, 2*c2, ...]
                deriv_coeffs = np.arange(1, 2*k+1) * lag_coeffs[k][1:]
                A[row_gp, k] = np.polyval(deriv_coeffs[::-1], z)
            else:
                A[row_gp, k] = 0.0
        b[row_g] = 0.0
        b[row_gp] = 0.0
        
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    g_coeffs = np.zeros(51)
    for k in range(26):
        g_coeffs += c[k] * lag_coeffs[k]
        
    D = np.array([1.0])
    for z in zs:
        quad = np.array([1.0, -2.0*z, z*z])
        D = np.convolve(D, quad)
    D = np.convolve(D, np.array([0.0, 1.0]))
    
    g_poly = g_coeffs[::-1]
    D_poly = D[::-1]
    
    try:
        q, r = np.polydiv(g_poly, D_poly)
    except Exception:
        return 1e10
        
    # Check remainder
    if len(r) > 0 and np.max(np.abs(r)) > 1e-4:
        # Division not exact, numerical error
        # We can still try to find roots of q, but it's risky
        pass
        
    roots = np.roots(q)
    real_roots = roots[np.abs(roots.imag) < 1e-8].real
    if len(real_roots) == 0:
        return 0.0
        
    max_r = np.max(real_roots)
    return max_r if max_r > 0 else 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 590
    
    n = 12
    alpha = -0.5
    
    x_init, _ = roots_genlaguerre(n, alpha)
    z_init = sorted([float(z) for z in x_init])
    
    best_zs = z_init
    best_score = _surrogate_score(best_zs)
    
    def objective(zs):
        if time.time() > deadline:
            return best_score
        if any(z <= 0 for z in zs):
            return 1e10
        if len(set(zs)) != len(zs):
            return 1e10
        if not all(np.isfinite(z) for z in zs):
            return 1e10
        zs_sorted = sorted(zs)
        return _surrogate_score(zs_sorted)
        
    try:
        result = minimize(objective, z_init, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False})
        if result.fun < best_score:
            best_zs = sorted([float(z) for z in result.x])
            best_score = result.fun
    except Exception:
        pass
        
    if len(best_zs) != 12 or any(z <= 0 or not np.isfinite(z) for z in best_zs):
        best_zs = z_init
        
    return best_zs
