import numpy as np
from scipy.optimize import minimize, basinhopping
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
            
        # Compute eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(J)
        except np.linalg.LinAlgError:
            return 1e10
            
        # Check if all eigenvalues are positive
        if np.any(eigvals <= 1e-6):
            # Penalize non-positive roots
            min_eig = np.min(eigvals)
            return 1e10 - min_eig
            
        # Compute norm squared
        norm_sq = compute_norm_sq_from_roots(eigvals)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Helper to get Jacobi params from roots
    def roots_to_jacobi(roots):
        # Sort roots
        roots = np.sort(roots)
        # Use roots as diagonal entries, small off-diagonals
        a = roots.copy()
        d = np.ones(m - 1) * 0.1
        return np.concatenate([a, d])

    # Seed 1: Geometric sequence
    seed_roots_1 = 4.0 * (2.0 ** np.arange(m))
    
    # Seed 2: Roots of Laguerre Polynomial L_12^(1)(x)
    from scipy.special import roots_genlaguerre
    try:
        x_lag, w_lag = roots_genlaguerre(m, 1)
        seed_roots_2 = x_lag
    except:
        seed_roots_2 = seed_roots_1

    # Prepare initial parameters
    init_params_1 = roots_to_jacobi(seed_roots_1)
    init_params_2 = roots_to_jacobi(seed_roots_2)
    
    # Bounds for optimization
    # Allow d to be negative to cover more space, but penalize if eigenvalues are not positive
    bounds = [(-1000, 10000)] * m + [(-1000, 10000)] * (m - 1)
    
    best_val = 1e10
    best_params = init_params_1
    
    # Evaluate initial costs
    val1 = cost_func_jacobi(init_params_1)
    val2 = cost_func_jacobi(init_params_2)
    
    if val1 < best_val:
        best_val = val1
        best_params = init_params_1
    if val2 < best_val:
        best_val = val2
        best_params = init_params_2
        
    # Optimization Strategy:
    # 1. Local polish from best seed
    # 2. Basin Hopping
    
    # 1. Local polish
    if time.time() < deadline - 20:
        try:
            res = minimize(cost_func_jacobi, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-12})
            if res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # 2. Basin Hopping
    if time.time() < deadline - 30:
        def basin_hopping_cost(params):
            return cost_func_jacobi(params)
            
        try:
            bh_res = basinhopping(
                basin_hopping_cost, 
                best_params, 
                niter=50, 
                stepsize=0.5, 
                mininterval=1, 
                maxinterval=10, 
                seed=42,
                disp=False
            )
            if bh_res.fun < best_val:
                best_val = bh_res.fun
                best_params = bh_res.x
        except Exception:
            pass

    # 3. Final high-precision polish
    if time.time() < deadline - 5:
        try:
            res2 = minimize(cost_func_jacobi, best_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-15})
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
        roots = seed_roots_1
        
    # Ensure roots are positive
    if np.any(roots <= 0):
        roots = seed_roots_1
    else:
        roots = np.maximum(roots, 1e-6)
        
    return list(roots)
