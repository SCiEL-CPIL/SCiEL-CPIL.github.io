import numpy as np
import scipy.optimize
import scipy.special
import time
import math

_START = time.time()
_DEADLINE = 630.0

def _objective(zs):
    # zs is a numpy array of 12 floats
    # Check constraints
    if np.any(zs <= 0):
        return 1e10
    # Check distinctness
    diffs = np.abs(np.subtract.outer(zs, zs))
    np.fill_diagonal(diffs, 1)
    if np.any(diffs < 1e-5):
        return 1e10 + 1e5 / np.min(diffs)
    
    # Sort for consistency (though objective is symmetric, sorting helps with potential issues)
    # Actually, the system is not symmetric in z order, but the set is.
    # The linear system rows are permuted if we permute z. The solution c is the same.
    # So we can sort z to avoid numerical issues? No, keep as is.
    
    n = 12
    alpha = -0.5
    
    # Precompute Laguerre polynomials and derivatives at z
    # L_{2k}(z) and L'_{2k}(z)
    # L'_{2k}^{(alpha)}(x) = - L_{2k-1}^{(alpha+1)}(x)
    
    # Matrix A (26x26)
    # Row 0: g(0) = 0
    # Row 1: g'(0) = 1
    # Rows 2..25: g(z_i) = 0, g'(z_i) = 0
    
    A = np.zeros((26, 26))
    b = np.zeros(26)
    
    # Laguerre at 0
    # L_n^{(alpha)}(0) = binom(n+alpha, n)
    # L'_n^{(alpha)}(0) = - binom(n-1+alpha+1, n-1) = - binom(n+alpha, n-1)
    
    # k=0..25
    for k in range(26):
        n_lag = 2 * k
        # Value at 0
        # binom(n+alpha, n)
        # Use gammaln for stability? n is small (50), direct binom is fine.
        val_0 = scipy.special.binom(n_lag + alpha, n_lag)
        # Derivative at 0
        # - binom(n_lag + alpha, n_lag - 1) if n_lag > 0 else 0
        if n_lag > 0:
            dval_0 = -scipy.special.binom(n_lag + alpha, n_lag - 1)
        else:
            dval_0 = 0.0
            
        A[0, k] = val_0
        A[1, k] = dval_0
        
        # Values at z_i
        for i in range(n):
            z = zs[i]
            # L_{2k}^{(alpha)}(z)
            val_z = scipy.special.eval_genlaguerre(n_lag, alpha, z)
            # L'_{2k}^{(alpha)}(z) = - L_{2k-1}^{(alpha+1)}(z)
            if n_lag > 0:
                dval_z = -scipy.special.eval_genlaguerre(n_lag - 1, alpha + 1, z)
            else:
                dval_z = 0.0
                
            A[2 + 2*i, k] = val_z
            A[2 + 2*i + 1, k] = dval_z
            
    b[1] = 1.0
    # Other b are 0
    
    try:
        c = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return 1e10
        
    # Compute monomial coefficients of g(x)
    # g(x) = sum c_k L_{2k}(x)
    # L_n^{(alpha)}(x) = sum_{m=0}^n (-1)^m binom(n+alpha, n-m) x^m / m!
    
    # Precompute basis coefficients
    # basis_coeffs[k, m] is coeff of x^m in L_{2k}(x)
    basis_coeffs = np.zeros((26, 51))
    for k in range(26):
        n_lag = 2 * k
        for m in range(n_lag + 1):
            # (-1)^m * binom(n_lag + alpha, n_lag - m) / factorial(m)
            coeff = ((-1)**m) * scipy.special.binom(n_lag + alpha, n_lag - m) / math.factorial(m)
            basis_coeffs[k, m] = coeff
            
    # g_coeffs = sum c_k * basis_coeffs[k, :]
    g_coeffs = np.dot(c, basis_coeffs)
    # g_coeffs[m] is coeff of x^m
    
    # Convert to numpy.poly1d format (descending order)
    g_poly = g_coeffs[::-1]
    
    # Construct D(x) = x * prod (x - z_i)^2
    # Start with [1] (x^0)
    d_poly = np.array([1.0])
    # Multiply by x
    d_poly = np.convolve(d_poly, [0.0, 1.0]) # x
    # Multiply by (x - z_i)^2 for each i
    for i in range(n):
        # (x - z)^2 = x^2 - 2z x + z^2
        # Coeffs: [1, -2z, z^2]
        factor = np.array([1.0, -2.0 * zs[i], zs[i]**2])
        d_poly = np.convolve(d_poly, factor)
        
    # d_poly is in ascending order?
    # np.convolve([1], [0,1]) -> [0,1] (x)
    # np.convolve([0,1], [1, -2z, z^2]) -> [0, 1, -2z, z^2] (x + x^2 - 2zx + z^2x) -> x^2 - 2zx + z^2?
    # Wait. np.convolve(a, v) is standard convolution.
    # If a=[1] (1), v=[0,1] (x), result [0,1] represents 0 + 1x.
    # If we treat index as power, [0,1] is x.
    # Let's stick to ascending order [c0, c1, ...]
    # g_coeffs is ascending.
    # d_poly is ascending.
    
    # Polynomial division g / D
    # numpy.polydiv expects descending order.
    g_desc = g_coeffs[::-1]
    d_desc = d_poly[::-1]
    
    # Remove leading zeros
    while len(g_desc) > 1 and g_desc[0] == 0: g_desc = g_desc[1:]
    while len(d_desc) > 1 and d_desc[0] == 0: d_desc = d_desc[1:]
    
    if len(d_desc) == 0 or d_desc[0] == 0:
        return 1e10
        
    q_desc, rem_desc = np.polydiv(g_desc, d_desc)
    
    # q_desc is descending order coeffs of q(x)
    # Find roots
    roots = np.roots(q_desc)
    
    # Filter real positive roots
    real_roots = roots[np.abs(roots.imag) < 1e-9]
    real_roots = real_roots.real
    pos_roots = real_roots[real_roots > 0]
    
    if len(pos_roots) == 0:
        return 0.0
        
    B = np.max(pos_roots)
    return B

def construct():
    best_zs = None
    best_score = float('inf')
    
    # Initial guesses
    starts = []
    # Geometric
    for base in [1.0, 1.2, 1.5]:
        starts.append(np.array([base * (1.1 ** i) for i in range(12)]))
    # Random
    rng = np.random.RandomState(42)
    for _ in range(5):
        starts.append(rng.uniform(0.5, 20, 12))
        
    for start in starts:
        if time.time() - _START > _DEADLINE - 30:
            break
            
        try:
            res = scipy.optimize.minimize(_objective, start, method='Nelder-Mead', options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_score:
                best_score = res.fun
                best_zs = res.x
        except Exception:
            continue
            
    if best_zs is None:
        # Fallback
        best_zs = np.array([4.0 * (i + 1) for i in range(12)])
        
    # Ensure distinct and positive
    # If Nelder-Mead produced invalid points, fallback to seed
    if np.any(best_zs <= 0) or len(set(np.round(best_zs, 5))) < 12:
         best_zs = np.array([4.0 * (i + 1) for i in range(12)])
         
    return [float(z) for z in best_zs]
