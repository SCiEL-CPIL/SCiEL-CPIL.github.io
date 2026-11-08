import numpy as np
from scipy.optimize import differential_evolution, minimize, dual_annealing
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s kill

    m = 12

    # Precompute the transformation matrix from monomial basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    # Precompute the weight vector for norm calculation
    weight_vec = np.arange(m + 1) + 1  # (k+1) for k=0..m

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.dot(c**2, weight_vec)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def roots_from_y(y):
        """
        Transform unconstrained y to ordered positive roots.
        r_1 = exp(y_1)
        r_i = r_{i-1} + exp(y_i)
        """
        r = np.zeros(m)
        r[0] = math.exp(y[0])
        for i in range(1, m):
            r[i] = r[i-1] + math.exp(y[i])
        return r

    def cost_func(y):
        """
        Minimize -log(norm_sq) which is equivalent to maximizing norm_sq.
        """
        try:
            roots = roots_from_y(y)
            # Check for numerical issues (exp overflow)
            if np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
                return 1e10
            
            norm_sq = compute_norm_sq(roots)
            
            if norm_sq <= 1e-10:
                return 1e10
            
            return -math.log(norm_sq)
        except Exception:
            return 1e10

    # Initial guess: geometric sequence
    # r_i = 4 * 2^i
    # y_1 = log(4)
    # y_i = log(r_i - r_{i-1}) = log(4*2^{i-1}) for i>1
    seed_y = np.zeros(m)
    seed_y[0] = math.log(4.0)
    for i in range(1, m):
        seed_y[i] = math.log(4.0 * (2.0 ** (i-1)))
        
    best_y = None
    best_val = 1e10

    # Strategy 1: Dual Annealing
    # Dual annealing is often robust for this kind of smooth, multimodal problem
    if time.time() - start_time < deadline - 300:
        try:
            da_result = dual_annealing(
                cost_func, 
                x0=seed_y, 
                seed=42,
                maxiter=1000,
                restarts=50,
                ac=None, # Default annealing
                pcvr=None, # Default
                x0_strategy='random'
            )
            if da_result.fun < best_val:
                best_val = da_result.fun
                best_y = da_result.x
        except Exception as e:
            pass

    # Strategy 2: Differential Evolution (multiple seeds)
    # Use tight bounds for y to keep roots in a reasonable range
    # Roots from 1 to 10000 -> y roughly in [-5, 10]
    bounds = [(-5.0, 10.0)] * m
    
    seeds = [42, 123, 456]
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 100:  # Leave time for polish
            break
        
        try:
            de_result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-10, 
                polish=False, # We do our own polish
                init='sobol'
            )
            if de_result.fun < best_val:
                best_val = de_result.fun
                best_y = de_result.x
        except Exception:
            pass

    # Fallback to seed if no optimization succeeded
    if best_y is None:
        best_y = seed_y
        best_val = cost_func(best_y)

    # Final High-Precision Polish
    if time.time() - start_time < deadline - 5:
        try:
            # Use L-BFGS-B or BFGS. Since y is unconstrained, BFGS is fine.
            # L-BFGS-B is also fine.
            polish_result = minimize(
                cost_func, 
                best_y, 
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_y = polish_result.x
        except Exception:
            pass

    # Convert best_y back to roots
    best_roots = roots_from_y(best_y)
    
    # Ensure roots are positive and sorted (they should be by construction)
    best_roots = np.maximum(best_roots, 1e-6)
    
    return list(best_roots)

