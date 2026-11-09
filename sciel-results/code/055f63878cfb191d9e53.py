import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import eval_genlaguerre
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 580  # 20s safety margin before 600s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # P(x) = sum c_k L_k^{(1)}(x)
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((13, 13))  # A[k, j] = coeff of x^j in L_k^{(1)}(x)
    for k in range(13):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = (A^T)^{-1} p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        # P(x) = prod(x - r_i)
        # np.poly returns descending coeffs: [1, c1, ..., cm]
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1] # ascending
        
        # c = A_T_inv @ p
        c = A_T_inv @ p
        
        # Norm squared: <L_k, L_l> = delta_kl * (k+1) for alpha=1
        # ||P||^2 = sum c_k^2 * (k+1)
        norm_sq = np.sum(c**2 * (np.arange(13) + 1))
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_from_roots(roots):
        """
        Compute cost (1/norm_sq) from a given array of 12 positive roots.
        """
        if len(roots) != 12:
            return 1e10
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
        norm_sq = compute_norm_sq(roots)
        if norm_sq <= 1e-10:
            return 1e10
        return 1.0 / norm_sq

    # PD matrix parameterization: D = D0 + X.T @ X
    # D is 12x12 SPD, roots are eigenvalues of D
    # X is 6x12, so X.T @ X is 12x12 PSD
    # D0 is a fixed 12x12 SPD baseline
    
    # Choose D0 as a diagonal matrix with reasonable spread
    # This gives a good starting point for the eigenvalue distribution
    d0_diag = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0])
    D0 = np.diag(d0_diag)
    
    def cost_pd(X_flat):
        """
        Cost function for PD parameterization.
        X_flat: flattened 6x12 matrix (72 elements)
        """
        X = X_flat.reshape(6, 12)
        # D = D0 + X.T @ X
        D = D0 + X.T @ X
        # Get eigenvalues (these are the roots)
        try:
            eigvals = np.linalg.eigvalsh(D)
        except np.linalg.LinAlgError:
            return 1e10
        
        if np.any(eigvals <= 0) or np.any(np.isnan(eigvals)) or np.any(np.isinf(eigvals)):
            return 1e10
            
        return cost_from_roots(eigvals)

    best_roots = None
    best_val = 1e10
    best_X_flat = None

    # Phase 1: PD matrix optimization using differential evolution
    # 72-dimensional search space
    # Use bounds that allow reasonable exploration
    bound = 5.0
    bounds_pd = [(-bound, bound)] * 72
    
    seeds_pd = [42, 123, 456]
    
    for seed in seeds_pd:
        elapsed = time.time() - start_time
        if elapsed > deadline - 60:
            break
        
        try:
            result = differential_evolution(
                cost_pd, 
                bounds_pd, 
                seed=seed, 
                maxiter=80, 
                popsize=12, 
                tol=1e-6, 
                polish=True,
                init='sobol',
                mutation=(0.3, 1.0),
                recombination=0.7
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_X_flat = result.x
                # Extract roots
                X = best_X_flat.reshape(6, 12)
                D = D0 + X.T @ X
                best_roots = np.linalg.eigvalsh(D)
                
        except Exception:
            pass

    # Phase 2: If PD search didn't find good results, try log-space as fallback
    if best_roots is None or time.time() - start_time < deadline - 90:
        def cost_func(log_roots):
            roots = np.exp(log_roots)
            if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
                return 1e10
            norm_sq = compute_norm_sq(roots)
            if norm_sq <= 1e-10:
                return 1e10
            return 1.0 / norm_sq

        seeds_log = [789, 101, 202]
        
        for seed in seeds_log:
            elapsed = time.time() - start_time
            if elapsed > deadline - 30:
                break
            
            try:
                result = differential_evolution(
                    cost_func, 
                    [(-3.0, 9.0)] * 12, 
                    seed=seed, 
                    maxiter=100, 
                    popsize=12, 
                    tol=1e-8, 
                    polish=False,
                    init='sobol'
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_roots = np.exp(result.x)
                    
            except Exception:
                pass

    # Fallback / Initial guess
    if best_roots is None:
        seed_roots = 4.0 * (2.0 ** np.arange(12))
        best_roots = seed_roots
        best_val = cost_from_roots(best_roots)

    # Final high-precision polish in PD space if we have a PD solution
    if best_X_flat is not None and time.time() - start_time < deadline - 10:
        try:
            polish_result = minimize(
                cost_pd, best_X_flat, method='L-BFGS-B',
                bounds=bounds_pd, options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_X_flat = polish_result.x
                X = best_X_flat.reshape(6, 12)
                D = D0 + X.T @ X
                best_roots = np.linalg.eigvalsh(D)
        except Exception:
            pass

    # If we have roots from log-space but not PD, polish in log-space
    elif best_roots is not None and best_X_flat is None and time.time() - start_time < deadline - 10:
        try:
            log_roots = np.log(np.maximum(best_roots, 1e-6))
            polish_result = minimize(
                cost_func, log_roots, method='L-BFGS-B',
                bounds=[(-3.0, 9.0)] * 12, options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-14}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_roots = np.exp(polish_result.x)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
