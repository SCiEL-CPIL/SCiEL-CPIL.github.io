import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 590  # 40s safety margin before 630s limit

    m = 12
    n = 1

    # Precompute the transformation matrix from standard basis to Laguerre basis
    # L_k^{(1)}(x) = sum_{j=0}^k A[k,j] x^j
    # A[k, j] = (-1)^j * binom(k+1, j+1) / j!
    
    A = np.zeros((m + 1, m + 1))
    for k in range(m + 1):
        for j in range(k + 1):
            A[k, j] = ((-1)**j) * math.comb(k+1, j+1) / math.factorial(j)
    
    # Solve for c given p. c = A_T_inv @ p
    try:
        A_T_inv = np.linalg.inv(A.T)
    except np.linalg.LinAlgError:
        A_T_inv = np.linalg.pinv(A.T)
    
    weight_vec = np.arange(m + 1) + 1  # (k+1) for k=0..m

    def compute_norm_sq(roots):
        """
        Compute the squared norm of the polynomial in the Laguerre basis.
        roots: array of 12 positive numbers.
        """
        # Build polynomial coefficients in standard basis (ascending)
        coeffs_desc = np.poly(roots)
        p = coeffs_desc[::-1]  # ascending
        
        c = A_T_inv @ p
        norm_sq = np.dot(c**2, weight_vec)
        
        if norm_sq <= 1e-10:
            return 0.0
        return norm_sq

    def cost_func(log_roots):
        """
        Minimize 1/||P||^2.
        """
        roots = np.exp(log_roots)
        
        if np.any(roots <= 0) or np.any(np.isnan(roots)) or np.any(np.isinf(roots)):
            return 1e10
            
        norm_sq = compute_norm_sq(roots)
        
        if norm_sq <= 1e-10:
            return 1e10
            
        return 1.0 / norm_sq

    # Seed: geometric sequence
    seed_log_roots = np.log(4.0) + np.arange(12) * np.log(2.0)
    
    # Bounds for log_roots
    bounds = [(-3.0, 11.0)] * 12

    best_roots = None
    best_val = 1e10
    best_log_roots = None

    # Phase 1: Multiple DE runs with different settings
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606]
    
    for seed in seeds:
        elapsed = time.time() - start_time
        if elapsed > deadline - 80:
            break
        
        try:
            result = differential_evolution(
                cost_func, 
                bounds, 
                seed=seed, 
                maxiter=150, 
                popsize=15, 
                tol=1e-8, 
                polish=True,
                init='sobol',
                mutation=(0.5, 1.0),
                recombination=0.8
            )
            
            if result.fun < best_val:
                best_val = result.fun
                best_log_roots = result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Fallback baseline
    seed_roots = 4.0 * (2.0 ** np.arange(12))
    seed_val = cost_func(seed_log_roots)
    
    if best_roots is None or seed_val < best_val:
        best_val = seed_val
        best_roots = seed_roots.copy()
        best_log_roots = seed_log_roots.copy()

    # Phase 2: Basin hopping from the best point found
    if time.time() - start_time < deadline - 30 and best_log_roots is not None:
        try:
            def local_minimize(x):
                res = minimize(cost_func, x, method='L-BFGS-B', 
                              bounds=bounds, options={'maxiter': 200, 'ftol': 1e-12})
                return res.fun, res.x, {}
            
            bh_result = basinhopping(
                local_minimize, 
                best_log_roots, 
                niter=20, 
                T=1.0, 
                stepsize=0.5,
                minimizer_kwargs={'bounds': bounds},
                seed=42
            )
            
            if bh_result.fun < best_val:
                best_val = bh_result.fun
                best_log_roots = bh_result.x.copy()
                best_roots = np.exp(best_log_roots)
                
        except Exception:
            pass

    # Phase 3: Additional DE runs from perturbed best point
    if time.time() - start_time < deadline - 20 and best_log_roots is not None:
        for trial in range(3):
            elapsed = time.time() - start_time
            if elapsed > deadline - 15:
                break
            try:
                # Perturb the best solution
                perturbation = np.random.RandomState(trial + 1000).normal(0, 0.5, 12)
                perturbed = best_log_roots + perturbation
                perturbed = np.clip(perturbed, -3.0, 11.0)
                
                result = minimize(
                    cost_func, perturbed, method='L-BFGS-B',
                    bounds=bounds, options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-14}
                )
                
                if result.fun < best_val:
                    best_val = result.fun
                    best_log_roots = result.x.copy()
                    best_roots = np.exp(best_log_roots)
                    
            except Exception:
                pass

    # Phase 4: Final high-precision polish
    if time.time() - start_time < deadline - 5 and best_log_roots is not None:
        try:
            polish_result = minimize(
                cost_func, best_log_roots, method='L-BFGS-B',
                bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-16, 'gtol': 1e-16}
            )
            if polish_result.fun < best_val:
                best_val = polish_result.fun
                best_log_roots = polish_result.x.copy()
                best_roots = np.exp(best_log_roots)
        except Exception:
            pass

    # Ensure roots are positive
    if best_roots is not None:
        best_roots = np.maximum(best_roots, 1e-6)
        return list(best_roots)
    
    # Absolute fallback
    return [4.0 * (2.0 ** i) for i in range(12)]
