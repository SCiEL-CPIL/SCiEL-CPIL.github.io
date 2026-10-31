"""Bootstrap DIRECT solution: optimize Hermite uncertainty bound with L-BFGS-B."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import factorial
from math import sqrt, pi

def _get_hermite_values():
    """Precompute H_n(0) and norms for n in {0, 4, 8, 12}."""
    idx = [0, 4, 8, 12]
    h0 = np.array([(-1)**(n//2) * factorial(n//2) / factorial(n) * (2**(n//2)) for n in idx])
    # Note: H_{2k}(0) = (-1)^k (2k)! / k! * 2^k? 
    # Formula: H_n(x) = n! sum_{m=0}^{floor(n/2)} (-1)^m (2x)^(n-2m) / (m! (n-2m)!)
    # H_n(0) = 0 for odd n.
    # For even n=2k: H_{2k}(0) = (-1)^k (2k)! / k! * 2^k? 
    # Let's check n=2. H_2(x) = 4x^2 - 2. H_2(0) = -2.
    # Formula: (-1)^1 * 2! / 1! * 2^1 = -1 * 2 * 2 = -4. Mismatch.
    # Correct formula: H_{2k}(0) = (-1)^k \frac{(2k)!}{k!}.
    # Check n=2 (k=1): (-1)^1 * 2! / 1! = -2. Correct.
    # Check n=4 (k=2): (-1)^2 * 4! / 2! = 12. Correct.
    # Check n=8 (k=4): (-1)^4 * 8! / 4! = 40320 / 24 = 1680. Correct.
    # Check n=12 (k=6): (-1)^6 * 12! / 6! = 479001600 / 720 = 665280.
    # Wait, previous thought said 103950. Let's recompute.
    # 12! = 479,001,600. 6! = 720. 479001600 / 720 = 665,280.
    # Let's re-verify the previous program's value or standard tables.
    # H_12(0) = 665280.
    
    h0 = np.array([
        1, 
        12, 
        1680, 
        665280
    ])
    # Signs:
    # H_0(0) = 1
    # H_4(0) = 12
    # H_8(0) = 1680
    # H_12(0) = 665280
    # Wait, (-1)^k.
    # k=0: 1
    # k=2: 1
    # k=4: 1
    # k=6: 1
    # So all positive?
    # H_2(0) = -2. k=1. (-1)^1 = -1.
    # H_4(0) = 12. k=2. (-1)^2 = 1.
    # H_8(0) = 1680. k=4. (-1)^4 = 1.
    # H_12(0) = 665280. k=6. (-1)^6 = 1.
    
    norms = np.array([sqrt(2**n * factorial(n) * sqrt(pi)) for n in idx])
    return idx, h0, norms

def _compute_objective(coeffs_free):
    """
    coeffs_free: [c0, c4, c8]
    Returns the upper bound on A(f)A(F(f)).
    """
    idx, h0, norms = _get_hermite_values()
    c0, c4, c8 = coeffs_free
    # c12 from constraint c0*h0[0] + c4*h0[1] + c8*h0[2] + c12*h0[3] = 0
    c12 = -(c0 * h0[0] + c4 * h0[1] + c8 * h0[2]) / h0[3]
    c = np.array([c0, c4, c8, c12])
    
    # Build matrices X and P
    # M_X[i,j] = <u_i, x^2 u_j>
    # M_P[i,j] = <u_i, u_j''>
    # Using derived formulas:
    # C_j = norm_j^2 = 2^j j! sqrt(pi)
    # M_X[i,j] = C_j [ delta(i, j+2)*(j+2)(j+1) + delta(i, j)*(j+0.5) + delta(i, j-2)*j(j-1) ]
    # M_P[i,j] = C_j [ delta(i, j+2)*(j+2)(j+1) - delta(i, j)*(j+0.5) + delta(i, j-2)*j(j-1) ]
    
    C = norms**2
    M_X = np.zeros((4, 4))
    M_P = np.zeros((4, 4))
    
    for j in range(4):
        n_j = idx[j]
        # Diagonal
        M_X[j, j] = C[j] * (n_j + 0.5)
        M_P[j, j] = -C[j] * (n_j + 0.5)
        
        # Off-diagonal j+2
        if j + 2 < 4:
            i = j + 2
            val = C[j] * (n_j + 2) * (n_j + 1)
            M_X[i, j] = val
            M_X[j, i] = val # Symmetric
            M_P[i, j] = val
            M_P[j, i] = val
            
        # Off-diagonal j-2
        if j - 2 >= 0:
            i = j - 2
            val = C[j] * n_j * (n_j - 1)
            M_X[i, j] = val
            M_X[j, i] = val
            M_P[i, j] = val
            M_P[j, i] = val
            
    # N = <g, g> = sum c_k^2 C_k
    N = np.sum(c**2 * C)
    
    if N <= 1e-12:
        return 1e10
        
    X_val = c @ M_X @ c
    P_val = c @ M_P @ c
    
    # Variance
    var_x = X_val / N
    var_p = P_val / N
    
    if var_x < 0 or var_p < 0:
        return 1e10
        
    return sqrt(var_x * var_p)

def construct():
    t0 = time.time()
    deadline = 180.0
    
    # Initial guess
    x0 = np.array([1.0, -0.1, 0.01])
    
    # Optimization
    result = minimize(_compute_objective, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12})
    
    if time.time() - t0 > deadline - 10:
        return np.array([1.0, 0.0, 0.0]) # Fallback
        
    best_x = result.x
    
    # Map to a0, a4, a8
    # a_k = c_k / norm_k
    idx, h0, norms = _get_hermite_values()
    a0 = best_x[0] / norms[0]
    a4 = best_x[1] / norms[1]
    a8 = best_x[2] / norms[2]
    
    return np.array([a0, a4, a8])
# EVOLVE-BLOCK-END
