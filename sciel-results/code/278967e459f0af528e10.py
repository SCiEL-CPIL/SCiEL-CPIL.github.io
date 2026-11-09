import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin before 600s limit

    m = 12
    max_deg = 2 * m

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^(1)(x)
    # L_k^(1)(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((max_deg + 1, max_deg + 1))
    for k in range(max_deg + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    weights = np.arange(max_deg + 1) + 1

    def compute_norm_sq_from_roots(roots):
        """
        Compute the squared norm of P(x) = prod (x - r_i)^2 in the Laguerre L_k^(1) basis.
        """
        # Build monic polynomial Q(x) = prod (x - r_i)
        # np.poly returns descending coefficients
        coeffs_Q_desc = np.poly(roots)
        
        # P(x) = Q(x)^2
        coeffs_P_desc = np.convolve(coeffs_Q_desc, coeffs_Q_desc)
        
        # Convert to ascending order
        p = coeffs_P_desc[::-1]
        
        # Pad if necessary
        if len(p) < max_deg + 1:
            p = np.pad(p, (0, max_deg + 1 - len(p)))
        else:
            p = p[:max_deg + 1]
            
        # Solve for Laguerre coefficients
        c = A_T_inv @ p
        
        # Norm squared
        norm_sq = np.sum(c**2 * weights)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func_jacobi(params):
        """
        Minimize 1/||P||^2.
        params: [a_0, ..., a_11, d_0, ..., d_10]
        """
        # Check time
        if time.time() > deadline:
            return 1e10
            
        a = params[:m]
        d = params[m:]
        
        # Construct Jacobi matrix
        J = np.zeros((m, m))
        for i in range(m):
            J[i, i] = a[i]
        for i in range(m - 1):
            J[i, i+1] = d[i]
            J[i+1, i] = d[i]
            
        # Ensure d_i > 0 (handled by bounds, but check just in case)
        if np.any(d <= 0):
            return 1e10
            
        # Compute eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(J)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Check if all eigenvalues are positive
        if np.any(eigvals <= 1e-6):
            # Penalize non-positive roots
            # The closer to 0 or negative, the higher the penalty
            min_eig = np.min(eigvals)
            return 1e10 - min_eig  # Or just 1e10
            
        # Compute norm squared
        norm_sq = compute_norm_sq_from_roots(eigvals)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence a=4, q=2
    seed_roots = 4.0 * (2.0 ** np.arange(m))
    
    # Initialize Jacobi matrix parameters from seed roots
    # We can start with a diagonal matrix (d=0) but that's a singular point in the manifold of irreducible Jacobi matrices.
    # Better to start with small d's.
    init_a = seed_roots.copy()
    init_d = np.ones(m - 1) * 0.1  # Small positive off-diagonals
    init_params = np.concatenate([init_a, init_d])
    
    # Bounds for optimization
    # Roots range from 4 to 8192. Allow some flexibility.
    # a_i can be negative in the matrix, as long as eigenvalues are positive.
    # But to keep it simple and stable, let's bound a_i loosely.
    bounds_a = [(-1000, 10000)] * m
    bounds_d = [(1e-6, 1000)] * (m - 1)
    bounds = bounds_a + bounds_d
    
    best_val = 1e10
    best_params = init_params
    
    # Evaluate initial cost
    initial_val = cost_func_jacobi(init_params)
    if initial_val < best_val:
        best_val = initial_val
        best_params = init_params
        
    # Optimization Strategy:
    # 1. Local polish from seed.
    # 2. Differential Evolution on Jacobi parameters (if time permits).
    # 3. Basin hopping or multiple restarts.
    
    # 1. Local polish
    if time.time() < deadline - 10:
        try:
            res = minimize(cost_func_jacobi, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 2. Differential Evolution on Jacobi parameters
    # This is a 23D problem. DE might be slow but could find global minima.
    # We have ~500s left.
    if time.time() < deadline - 60:
        # Use a subset of the domain for DE to keep it manageable
        # The eigenvalues must be positive. The matrix entries can vary.
        # Let's try DE with a smaller population and iterations.
        try:
            # We need to be careful with bounds for DE.
            # If a_i is too large, eigenvalues might be large.
            # Let's restrict DE bounds to be tighter than the general bounds.
            de_bounds = [(-100, 1000)] * m + [(0.01, 100)] * (m - 1)
            
            # Run DE
            de_res = differential_evolution(
                cost_func_jacobi, 
                de_bounds, 
                seed=42, 
                maxiter=50, 
                popsize=10, 
                tol=1e-8, 
                polish=True
            )
            
            if de_res.fun < best_val:
                best_val = de_res.fun
                best_params = de_res.x
        except Exception:
            pass
            
    # 3. Additional local polish from best found so far
    if time.time() < deadline - 5:
        try:
            res2 = minimize(cost_func_jacobi, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-18, 'gtol': 1e-18})
            if res2.fun < best_val:
                best_val = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    # Extract roots from best Jacobi matrix
    a = best_params[:m]
    d = best_params[m:]
    J = np.zeros((m, m))
    for i in range(m):
        J[i, i] = a[i]
    for i in range(m - 1):
        J[i, i+1] = d[i]
        J[i+1, i] = d[i]
        
    try:
        roots = np.linalg.eigvalsh(J)
    except np.linalg.LinAlgError:
        roots = seed_roots
        
    # Ensure roots are positive
    if np.any(roots <= 0):
        # Fallback to seed if optimization failed to produce valid roots
        roots = seed_roots
    else:
        roots = np.maximum(roots, 1e-6)
        
    return list(roots)
