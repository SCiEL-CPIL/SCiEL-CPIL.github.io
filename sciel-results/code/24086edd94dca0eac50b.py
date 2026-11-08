import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import math

def _compute_bound(c):
    """
    c: array [c0, c4, c8, c12]
    Returns the uncertainty product Δx * Δp for f(x) = sum c_n H_n(x) exp(-x^2/2)
    Uses analytical matrix elements for Hermite functions.
    """
    degrees = np.array([0, 4, 8, 12])
    coeffs = c
    N = 4
    idx = degrees
    
    # Matrix for x: <psi_n|x|psi_m>
    M_x = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n = idx[i]
            m = idx[j]
            if n == m + 1:
                M_x[i, j] = np.sqrt(m + 1) / np.sqrt(2)
            elif n == m - 1:
                M_x[i, j] = np.sqrt(m) / np.sqrt(2)
                
    # Matrix for x^2: <psi_n|x^2|psi_m>
    M_x2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            n = idx[i]
            m = idx[j]
            if n == m + 2:
                M_x2[i, j] = 0.5 * np.sqrt((m + 1) * (m + 2))
            elif n == m:
                M_x2[i, j] = m + 0.5
            elif n == m - 2:
                M_x2[i, j] = 0.5 * np.sqrt(m * (m - 1))
                
    # Matrix for p: <psi_n|p|psi_m>
    # p is Hermitian, elements are imaginary
    M_p = np.zeros((N, N), dtype=complex)
    for i in range(N):
        for j in range(N):
            n = idx[i]
            m = idx[j]
            if n == m - 1:
                M_p[i, j] = 1j * np.sqrt(m) / np.sqrt(2)
            elif n == m + 1:
                M_p[i, j] = -1j * np.sqrt(m + 1) / np.sqrt(2)
                
    # Matrix for p^2: <psi_n|p^2|psi_m> = (2m+1)delta_nm - <psi_n|x^2|psi_m>
    M_p2 = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                M_p2[i, j] = 2.0 * idx[j] + 1.0
            M_p2[i, j] -= M_x2[i, j]
            
    # Coefficients in orthonormal basis
    # f = sum c_k H_{d_k} e^{-x^2/2} = sum a_k psi_{d_k}
    # a_k = c_k * sqrt(2^{d_k} d_k! sqrt(pi))
    # We can drop the common sqrt(sqrt(pi)) factor.
    w = np.zeros(N)
    for k in range(N):
        fact = math.factorial(idx[k])
        w[k] = coeffs[k] * np.sqrt((2.0 ** idx[k]) * fact)
        
    norm_sq = np.dot(w, w)
    if norm_sq < 1e-15:
        return np.inf
        
    exp_x = np.dot(w, np.dot(M_x, w))
    exp_x2 = np.dot(w, np.dot(M_x2, w))
    exp_p = np.dot(w, np.dot(M_p, w)).real
    exp_p2 = np.dot(w, np.dot(M_p2, w))
    
    var_x = exp_x2 - exp_x**2
    var_p = exp_p2 - exp_p**2
    
    if var_x < 0: var_x = 0
    if var_p < 0: var_p = 0
    
    return np.sqrt(var_x) * np.sqrt(var_p)

def construct():
    start_time = time.time()
    deadline = 200.0
    
    def objective(params):
        c4, c8 = params
        c0 = 1.0
        c12 = -(c0 + 12.0*c4 + 1680.0*c8) / 3991680.0
        c = np.array([c0, c4, c8, c12])
        return _compute_bound(c)
    
    best_score = np.inf
    best_params = np.array([0.0, 0.0])
    
    if time.time() - start_time < deadline - 10:
        try:
            bounds = [(-1.0, 1.0), (-0.1, 0.1)]
            de_result = differential_evolution(
                objective, bounds, seed=42, maxiter=200, 
                popsize=20, tol=1e-10, polish=True,
                init='sobol'
            )
            if de_result.fun < best_score:
                best_score = de_result.fun
                best_params = de_result.x
        except Exception:
            pass
    
    if time.time() - start_time < deadline - 5 and np.isfinite(best_score):
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    return np.array([1.0, best_params[0], best_params[1]])
